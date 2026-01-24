#!/bin/bash

# Script to display the current GIT_REPO_REF and status for showroom deployments
# Usage:
#   ./show-showroom-branch.sh                    # Show current namespace
#   ./show-showroom-branch.sh --all              # Show all showroom namespaces
#   ./show-showroom-branch.sh -n <ns>            # Show specific namespace

set -e

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
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --all, -a           Show all showroom-* namespaces"
            echo "  -n, --namespace NS  Show specific namespace"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # Show current namespace"
            echo "  $0 --all                        # Show all showroom namespaces"
            echo "  $0 -n showroom-xyz-user1        # Show specific namespace"
            exit 0
            ;;
        *)
            echo "Error: Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Function to show info for a single namespace
show_namespace() {
    local ns=$1

    # Check if showroom deployment exists
    if ! oc get deployment showroom -n "$ns" &>/dev/null; then
        printf "%-40s %-25s %-10s %-10s\n" "$ns" "NO DEPLOYMENT" "-" "-"
        return
    fi

    # Get init container GIT_REPO_REF
    local init_branch
    init_branch=$(oc get deployment showroom -n "$ns" -o jsonpath='{.spec.template.spec.initContainers[0].env[?(@.name=="GIT_REPO_REF")].value}' 2>/dev/null || echo "N/A")

    # Get pod status
    local pod_info
    pod_info=$(oc get pods -n "$ns" -l app.kubernetes.io/name=showroom -o jsonpath='{.items[0].status.phase}:{.items[0].status.containerStatuses[*].ready}' 2>/dev/null || echo "N/A:N/A")

    local pod_status="${pod_info%%:*}"
    local ready_status="${pod_info##*:}"

    # Get ready count
    local ready_count="0/0"
    if [[ "$pod_status" != "N/A" ]]; then
        local total=$(echo "$ready_status" | wc -w | tr -d ' ')
        local ready=$(echo "$ready_status" | grep -o "true" | wc -l | tr -d ' ')
        ready_count="${ready}/${total}"
    fi

    # Get pod age
    local pod_age
    pod_age=$(oc get pods -n "$ns" -l app.kubernetes.io/name=showroom -o jsonpath='{.items[0].metadata.creationTimestamp}' 2>/dev/null || echo "")
    if [[ -n "$pod_age" && "$pod_age" != "" ]]; then
        # Calculate age in a simple format
        pod_age=$(oc get pods -n "$ns" -l app.kubernetes.io/name=showroom -o jsonpath='{range .items[0]}{.metadata.creationTimestamp}{end}' 2>/dev/null)
        # Get age from oc get pods output
        pod_age=$(oc get pods -n "$ns" -l app.kubernetes.io/name=showroom --no-headers 2>/dev/null | awk '{print $5}' | head -1)
    fi
    [[ -z "$pod_age" ]] && pod_age="N/A"

    printf "%-40s %-25s %-10s %-10s %s\n" "$ns" "$init_branch" "$pod_status" "$ready_count" "$pod_age"
}

# Print header
print_header() {
    echo ""
    printf "%-40s %-25s %-10s %-10s %s\n" "NAMESPACE" "GIT_REPO_REF" "STATUS" "READY" "AGE"
    printf "%-40s %-25s %-10s %-10s %s\n" "---------" "------------" "------" "-----" "---"
}

# Main logic
if [[ "$ALL_NAMESPACES" == true ]]; then
    NAMESPACES=$(oc get ns -o name | grep showroom | grep -v showroom-ai-assistant | sed 's|namespace/||' | sort)

    if [[ -z "$NAMESPACES" ]]; then
        echo "No showroom namespaces found"
        exit 1
    fi

    print_header
    for ns in $NAMESPACES; do
        show_namespace "$ns"
    done
    echo ""

elif [[ -n "$NAMESPACE" ]]; then
    print_header
    show_namespace "$NAMESPACE"
    echo ""

else
    # Use current namespace
    CURRENT_NS=$(oc project -q)
    print_header
    show_namespace "$CURRENT_NS"
    echo ""
fi
