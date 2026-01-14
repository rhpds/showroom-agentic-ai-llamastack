'use strict'

const fs = require('fs')
const path = require('path')

/**
 * Antora extension that exports AsciiDoc content with resolved attributes
 * for RAG (Retrieval Augmented Generation) purposes.
 *
 * This extension intercepts the Antora build process after attribute substitution
 * but before HTML rendering, giving you access to the "user-facing" content.
 */
module.exports.register = function ({ config }) {
  const logger = this.getLogger('rag-export-extension')

  // Default configuration
  const outputDir = config.outputDir || './rag-content'
  const enabled = config.enabled !== false

  if (!enabled) {
    logger.info('RAG export is disabled')
    return
  }

  let playbookOutputDir = null

  this.once('playbookBuilt', ({ playbook }) => {
    // Capture the output directory from the playbook
    playbookOutputDir = playbook.output?.dir
    logger.info(`Playbook output directory: ${playbookOutputDir}`)
  })

  this.on('documentsConverted', ({ contentCatalog, siteCatalog }) => {
    logger.info('Exporting documents with resolved attributes for RAG...')

    // Create output directory if it doesn't exist
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true })
    }

    // Get all page documents
    const pages = contentCatalog.findBy({ family: 'page' })
    let exportedCount = 0

    pages.forEach((page) => {
      // Skip special files if needed
      const filename = page.src.basename
      if (filename === 'ai-chatbot.adoc' || filename === 'nav.adoc' || filename === 'attrs-page.adoc') {
        return
      }

      try {
        // Access the Asciidoctor document object
        const doc = page.asciidoc

        if (!doc) {
          logger.warn(`No AsciiDoc document for ${page.src.relative}`)
          return
        }

        // Get title - Antora stores this in the page object
        const title = page.asciidoc.doctitle || page.title || page.src.stem

        // Get attributes - stored in the asciidoc attributes object
        const attributes = page.asciidoc.attributes || {}

        // Get the source with attribute substitution
        const resolvedContent = resolveAttributesInSource(page, attributes, logger)

        // Create metadata header
        const metadata = {
          title: title,
          component: page.src.component,
          version: page.src.version,
          module: page.src.module,
          originalPath: page.src.relative,
          url: page.pub?.url || '',
          relevantAttributes: extractRelevantAttributes(attributes)
        }

        // Combine metadata and content
        const output = `---
METADATA:
${JSON.stringify(metadata, null, 2)}
---

${resolvedContent}
`

        // Generate output filename
        const outputFilename = `${page.src.component}-${page.src.module}-${page.src.stem}.txt`
        const outputPath = path.join(outputDir, outputFilename)

        // Write to file
        fs.writeFileSync(outputPath, output, 'utf-8')
        exportedCount++

        logger.debug(`Exported: ${outputFilename}`)

      } catch (error) {
        logger.error(`Error exporting ${page.src.relative}: ${error.message}`)
        logger.error(error.stack)
      }
    })

    logger.info(`Successfully exported ${exportedCount} documents to ${outputDir}`)

    // Copy techdocs (PDFs) to the site output directory
    logger.info('Copying techdocs (PDFs) to site output...')
    copyTechdocs(contentCatalog, playbookOutputDir, logger)
  })
}

/**
 * Copy techdocs (PDFs) to the Antora site output directory
 * so they're accessible at /_/techdocs/filename.pdf
 */
function copyTechdocs(contentCatalog, siteDir, logger) {
  try {
    if (!siteDir) {
      logger.error('Could not determine site output directory - playbook output dir not available')
      return
    }

    // Create _/techdocs directory in site output
    const techdocsOutputDir = path.join(siteDir, '_', 'techdocs')
    if (!fs.existsSync(techdocsOutputDir)) {
      fs.mkdirSync(techdocsOutputDir, { recursive: true })
      logger.info(`Created techdocs output directory: ${techdocsOutputDir}`)
    }

    // Instead of relying on content catalog, directly scan the filesystem for PDFs
    // Get the content source location from any page in the catalog
    const pages = contentCatalog.findBy({ family: 'page' })
    if (pages.length === 0) {
      logger.warn('No pages found in content catalog, cannot determine source directory')
      return
    }

    const firstPage = pages[0]
    const worktree = firstPage.src.origin?.worktree
    const startPath = firstPage.src.origin?.startPath || ''

    if (!worktree) {
      logger.error('Could not determine content worktree path')
      return
    }

    // Build path to techdocs directory: worktree/startPath/modules/ROOT/assets/techdocs
    const techdocsSourceDir = path.join(worktree, startPath, 'modules', 'ROOT', 'assets', 'techdocs')

    logger.info(`Looking for PDFs in: ${techdocsSourceDir}`)

    if (!fs.existsSync(techdocsSourceDir)) {
      logger.warn(`Techdocs source directory not found: ${techdocsSourceDir}`)
      return
    }

    // Read all files in the techdocs directory
    const files = fs.readdirSync(techdocsSourceDir)
    const pdfFiles = files.filter(file => file.endsWith('.pdf'))

    logger.info(`Found ${pdfFiles.length} PDF files in techdocs directory`)

    let copiedCount = 0

    pdfFiles.forEach((pdfFile) => {
      try {
        const sourcePath = path.join(techdocsSourceDir, pdfFile)
        const destPath = path.join(techdocsOutputDir, pdfFile)

        logger.info(`Copying PDF: ${pdfFile}`)
        logger.debug(`  From: ${sourcePath}`)
        logger.debug(`  To: ${destPath}`)

        fs.copyFileSync(sourcePath, destPath)
        copiedCount++
        logger.info(`✓ Successfully copied: ${pdfFile}`)
      } catch (error) {
        logger.error(`Error copying PDF ${pdfFile}: ${error.message}`)
      }
    })

    logger.info(`Successfully copied ${copiedCount}/${pdfFiles.length} PDF files to ${techdocsOutputDir}`)
  } catch (error) {
    logger.error(`Error in copyTechdocs: ${error.message}`)
    logger.error(error.stack)
  }
}

/**
 * Resolve attributes in the original source
 * This reads the original .adoc file and manually substitutes attributes
 */
function resolveAttributesInSource(page, attributes, logger) {
  try {
    // Try to find the original source file
    // The page.src.origin.worktree gives us the base directory
    const worktree = page.src.origin?.worktree
    const startPath = page.src.origin?.startPath || ''

    if (!worktree) {
      logger.warn(`No worktree found for ${page.src.relative}, using HTML fallback`)
      return htmlToText(page.contents.toString('utf-8'))
    }

    // Build the full path to the source file
    // Antora structure: worktree/startPath/modules/MODULE/pages/file.adoc
    const modulePath = page.src.module || 'ROOT'
    const familyPlural = page.src.family === 'page' ? 'pages' : page.src.family + 's'
    const fullPath = path.join(worktree, startPath, 'modules', modulePath, familyPlural, page.src.relative)

    if (!fs.existsSync(fullPath)) {
      logger.warn(`Source file not found at ${fullPath}, using HTML fallback`)
      return htmlToText(page.contents.toString('utf-8'))
    }

    // Read the original source file
    let content = fs.readFileSync(fullPath, 'utf-8')

    // Substitute attributes
    // AsciiDoc attribute syntax is {attribute_name}
    Object.keys(attributes).forEach(key => {
      const value = attributes[key]
      if (typeof value === 'string') {
        // Replace all occurrences of {key} with the value
        const regex = new RegExp(`\\{${escapeRegex(key)}\\}`, 'g')
        content = content.replace(regex, value)
      }
    })

    return content
  } catch (error) {
    logger.warn(`Error reading source for ${page.src.relative}: ${error.message}, using HTML fallback`)
    // Fallback: return HTML converted to text
    const htmlContent = page.contents.toString('utf-8')
    return htmlToText(htmlContent)
  }
}

/**
 * Escape special regex characters in a string
 */
function escapeRegex(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/**
 * Simple HTML to text conversion (fallback)
 */
function htmlToText(html) {
  let text = html

  // Remove scripts and styles
  text = text.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
  text = text.replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '')

  // Replace common HTML entities
  text = text.replace(/&nbsp;/g, ' ')
  text = text.replace(/&#8217;/g, "'")
  text = text.replace(/&#8220;/g, '"')
  text = text.replace(/&#8221;/g, '"')
  text = text.replace(/&quot;/g, '"')
  text = text.replace(/&apos;/g, "'")
  text = text.replace(/&amp;/g, '&')
  text = text.replace(/&lt;/g, '<')
  text = text.replace(/&gt;/g, '>')

  // Remove HTML tags
  text = text.replace(/<[^>]+>/g, ' ')

  // Clean up whitespace
  text = text.replace(/\s+/g, ' ')
  text = text.replace(/\n\s*\n/g, '\n\n')

  return text.trim()
}

/**
 * Extract relevant attributes that were used in the document
 */
function extractRelevantAttributes(attributes) {
  const attrs = {}

  // Common workshop attributes
  const relevantAttrs = [
    'lab_name', 'guid', 'ssh_user', 'ssh_password', 'ssh_command',
    'release-version', 'workshop_title', 'assistant_name',
    'my_var', 'welcome_message'
  ]

  relevantAttrs.forEach(attr => {
    const value = attributes[attr]
    if (value !== undefined && typeof value === 'string') {
      attrs[attr] = value
    }
  })

  return attrs
}
