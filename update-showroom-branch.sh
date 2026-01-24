#!/bin/bash

# Script to update the GIT_REPO_REF for showroom deployments
# Usage:
#   ./update-showroom-branch.sh <branch-name>              # Update current namespace
#   ./update-showroom-branch.sh <branch-name> --all        # Update all showroom namespaces
#   ./update-showroom-branch.sh <branch-name> -n <ns>      # Update specific namespace

set -e

BRANCH=""
NAMESPACE=""
ALL_NAMESPACES=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all|-a)
            ALL_NAMESPACES=true
            shift
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 <branch-name> [options]"
            echo ""
            echo "Options:"
            echo "  --all, -a           Update all showroom-* namespaces"
            echo "  -n, --namespace NS  Update specific namespace"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 rhods-operator.3.2.0              # Update current namespace"
            echo "  $0 rhods-operator.3.2.0 --all        # Update all showroom namespaces"
            echo "  $0 main -n showroom-xyz-user1        # Update specific namespace"
            exit 0
            ;;
        *)
            if [[ -z "$BRANCH" ]]; then
                BRANCH="$1"
            else
                echo "Error: Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate branch parameter
if [[ -z "$BRANCH" ]]; then
    echo "Error: Branch name is required"
    echo "Usage: $0 <branch-name> [--all | -n <namespace>]"
    exit 1
fi

# Function to update a single namespace
update_namespace() {
    local ns=$1
    echo "Updating namespace: $ns"

    # Check if showroom deployment exists
    if ! oc get deployment showroom -n "$ns" &>/dev/null; then
        echo "  Warning: No showroom deployment found in $ns, skipping..."
        return
    fi

    # Get current init container env index for GIT_REPO_REF
    local env_index
    env_index=$(oc get deployment showroom -n "$ns" -o json | \
        jq '.spec.template.spec.initContainers[0].env | to_entries | map(select(.value.name == "GIT_REPO_REF")) | .[0].key')

    if [[ "$env_index" == "null" || -z "$env_index" ]]; then
        echo "  Warning: GIT_REPO_REF not found in init container for $ns, skipping..."
        return
    fi

    # Update init container (git-cloner)
    echo "  Patching init container git-cloner..."
    oc patch deployment showroom -n "$ns" --type='json' \
        -p="[{\"op\": \"replace\", \"path\": \"/spec/template/spec/initContainers/0/env/$env_index/value\", \"value\": \"$BRANCH\"}]"

    # Update main containers
    echo "  Updating main containers..."
    oc set env deployment/showroom GIT_REPO_REF="$BRANCH" -n "$ns" 2>/dev/null || true

    # Wait for rollout
    echo "  Waiting for rollout..."
    oc rollout status deployment/showroom -n "$ns" --timeout=180s

    echo "  Done: $ns updated to $BRANCH"
    echo ""
}

# Main logic
if [[ "$ALL_NAMESPACES" == true ]]; then
    echo "Updating all showroom namespaces to branch: $BRANCH"
    echo "================================================"

    NAMESPACES=$(oc get ns -o name | grep showroom | grep -v showroom-ai-assistant | sed 's|namespace/||')

    if [[ -z "$NAMESPACES" ]]; then
        echo "No showroom namespaces found"
        exit 1
    fi

    for ns in $NAMESPACES; do
        update_namespace "$ns"
    done

    echo "All namespaces updated successfully!"

elif [[ -n "$NAMESPACE" ]]; then
    echo "Updating namespace $NAMESPACE to branch: $BRANCH"
    echo "================================================"
    update_namespace "$NAMESPACE"

else
    # Use current namespace
    CURRENT_NS=$(oc project -q)
    echo "Updating current namespace ($CURRENT_NS) to branch: $BRANCH"
    echo "================================================"
    update_namespace "$CURRENT_NS"
fi

echo ""
echo "To verify, check the git-cloner logs:"
echo "  oc logs deployment/showroom -c git-cloner -n <namespace> | head -10"
