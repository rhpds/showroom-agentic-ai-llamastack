#!/usr/bin/env node

/**
 * Disable AI assistant extensions for GitHub Pages build
 * This script modifies default-site.yml to disable ai-assistant-build and rag-export extensions
 */

const fs = require('fs');
const yaml = require('js-yaml');

const configFile = 'default-site.yml';

try {
  // Read the YAML file
  const fileContents = fs.readFileSync(configFile, 'utf8');
  const config = yaml.load(fileContents);

  // Find and disable the extensions
  if (config.antora && config.antora.extensions) {
    config.antora.extensions.forEach(ext => {
      if (ext.id === 'ai-assistant-build' || ext.id === 'rag-export') {
        ext.enabled = false;
        console.log(`✓ Disabled extension: ${ext.id}`);
      }
    });
  }

  // Write the modified YAML back
  fs.writeFileSync(configFile, yaml.dump(config, { lineWidth: -1 }), 'utf8');
  console.log(`✓ Successfully updated ${configFile}`);

} catch (error) {
  console.error(`✗ Error modifying ${configFile}:`, error.message);
  process.exit(1);
}
