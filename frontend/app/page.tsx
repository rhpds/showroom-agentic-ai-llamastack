'use client'

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from '@/components/ui/shadcn-io/ai/conversation'
import { Loader } from '@/components/ui/shadcn-io/ai/loader'
import { Message, MessageAvatar, MessageContent } from '@/components/ui/shadcn-io/ai/message'
import { Reasoning, ReasoningTrigger, ReasoningContent } from '@/components/ui/shadcn-io/ai/reasoning'
import { Response } from '@/components/ui/shadcn-io/ai/response'
import { Sources, SourcesTrigger, SourcesContent, Source } from '@/components/ui/shadcn-io/ai/source'
import { Tool, ToolHeader, ToolContent, ToolInput, ToolOutput } from '@/components/ui/shadcn-io/ai/tool'
import {
  PromptInput,
  PromptInputModelSelect,
  PromptInputModelSelectContent,
  PromptInputModelSelectItem,
  PromptInputModelSelectTrigger,
  PromptInputModelSelectValue,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputToolbar,
  PromptInputTools,
} from '@/components/ui/shadcn-io/ai/prompt-input'
import { Button } from '@/components/ui/button'
import { MarkdownRenderer } from '@/components/ui/markdown-renderer'
import { RotateCcwIcon } from 'lucide-react'
import { nanoid } from 'nanoid'
import { type FormEventHandler, useCallback, useEffect, useState } from 'react'
import { flushSync } from 'react-dom'

type Source = {
  title: string
  url: string
  content_type: string
}

type ToolCall = {
  id: string
  name: string
  state: 'input-streaming' | 'input-available' | 'output-available' | 'output-error'
  input?: any
  output?: any
  errorText?: string
}

type ChatMessage = {
  id: string
  content: string
  role: 'user' | 'assistant'
  timestamp: Date
  isStreaming?: boolean
  sources?: Source[]
  reasoning?: string
  toolCalls?: Record<string, ToolCall>
}

type Agent = {
  id: string
  name: string
  description: string
}

// Helper function to convert file paths to URLs
function sourceToUrl(source: Source): string {
  const { url, content_type } = source

  // Both PDFs and workshop pages need to point to the showroom (not AI assistant)
  // Pattern: showroom-ai-assistant-<namespace>.<subdomain> -> showroom-<namespace>.<subdomain>
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname

    // For localhost, just use relative URL
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return url
    }

    // For OpenShift/production: replace 'showroom-ai-assistant' with 'showroom'
    const showroomHostname = hostname.replace('showroom-ai-assistant-', 'showroom-')
    const showroomUrl = `${window.location.protocol}//${showroomHostname}${url}`
    return showroomUrl
  }

  // Fallback for SSR
  return url
}

const STORAGE_KEY_MESSAGES = 'ai-assistant-messages'
const STORAGE_KEY_SELECTED_AGENT = 'ai-assistant-selected-agent'

export default function ChatPage() {
  // Initialize messages from localStorage or default welcome message
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    if (typeof window === 'undefined') return []

    try {
      const stored = localStorage.getItem(STORAGE_KEY_MESSAGES)
      if (stored) {
        const parsed = JSON.parse(stored)
        // Convert timestamp strings back to Date objects and clean file references
        return parsed.map((msg: any) => ({
          ...msg,
          content: msg.content ? msg.content.replace(/<\|file-[a-f0-9]+\|>/g, '') : msg.content,
          timestamp: msg.timestamp ? new Date(msg.timestamp) : new Date()
        }))
      }
    } catch (error) {
      console.error('Failed to load messages from localStorage:', error)
    }

    // Default welcome message
    return [{
      id: nanoid(),
      content: "Hello! I'm your workshop AI assistant. I can help you with questions about the workshop content, troubleshooting, and guidance. What would you like to know?",
      role: 'assistant',
      timestamp: new Date()
    }]
  })

  const [inputValue, setInputValue] = useState('')
  const [agents, setAgents] = useState<Agent[]>([])
  const [exampleQuestions, setExampleQuestions] = useState<string[]>([])
  const [selectedAgent, setSelectedAgent] = useState(() => {
    if (typeof window === 'undefined') return 'auto'
    return localStorage.getItem(STORAGE_KEY_SELECTED_AGENT) || 'auto'
  })
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null)
  const [previousResponseId, setPreviousResponseId] = useState<string | null>(null)

  // Save messages to localStorage whenever they change
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(STORAGE_KEY_MESSAGES, JSON.stringify(messages))
    }
  }, [messages])

  // Save selected agent to localStorage whenever it changes
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(STORAGE_KEY_SELECTED_AGENT, selectedAgent)
    }
  }, [selectedAgent])

  // Dynamically construct backend URL based on current hostname
  // Pattern: <anything>.<cluster-domain> -> showroom-ai-assistant-showroom-ai-assistant.<cluster-domain>
  const getBackendUrl = () => {
    if (typeof window === 'undefined') {
      return 'http://localhost:8001' // SSR fallback
    }

    const hostname = window.location.hostname

    // For localhost development
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return 'http://localhost:8001'
    }

    // For OpenShift/Kubernetes deployment
    // Replace everything up to the first dot with 'showroom-ai-assistant-showroom-ai-assistant'
    // Pattern: <service-namespace>.<cluster-domain> -> showroom-ai-assistant-showroom-ai-assistant.<cluster-domain>
    const firstDotIndex = hostname.indexOf('.')
    if (firstDotIndex !== -1) {
      // Take everything after the first dot (cluster domain)
      const clusterDomain = hostname.substring(firstDotIndex + 1)
      // Prepend showroom-ai-assistant-showroom-ai-assistant
      const backendHostname = `showroom-ai-assistant-showroom-ai-assistant.${clusterDomain}`
      return `${window.location.protocol}//${backendHostname}`
    }

    // Fallback
    return 'http://localhost:8001'
  }

  const backendUrl = getBackendUrl()

  // Fetch available agents on mount
  useEffect(() => {
    fetch(`${backendUrl}/api/agents`)
      .then(res => res.json())
      .then(data => {
        setAgents(data.agents || [])
      })
      .catch(err => console.error('Failed to fetch agents:', err))
  }, [backendUrl])

  // Fetch configuration (example questions) on mount
  useEffect(() => {
    fetch(`${backendUrl}/api/config`)
      .then(res => res.json())
      .then(data => {
        setExampleQuestions(data.example_questions || [])
      })
      .catch(err => console.error('Failed to fetch config:', err))
  }, [backendUrl])

  // Extracted chat submission logic
  const submitMessage = useCallback(
    async (messageText: string) => {
      if (!messageText.trim() || isStreaming) return

      // Add user message
      const userMessage: ChatMessage = {
        id: nanoid(),
        content: messageText.trim(),
        role: 'user',
        timestamp: new Date()
      }

      setMessages(prev => [...prev, userMessage])
      setIsStreaming(true)

      // Create empty assistant message for streaming
      const assistantMessageId = nanoid()
      const assistantMessage: ChatMessage = {
        id: assistantMessageId,
        content: '',
        role: 'assistant',
        timestamp: new Date(),
        isStreaming: true
      }

      setMessages(prev => [...prev, assistantMessage])
      setStreamingMessageId(assistantMessageId)

      try {
        const response = await fetch(`${backendUrl}/api/chat/stream`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: userMessage.content,
            agent_type: selectedAgent,
            include_mcp: true,
            page_context: null,
            previous_response_id: previousResponseId
          })
        })

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const reader = response.body?.getReader()
        const decoder = new TextDecoder()

        if (reader) {
          let buffer = ''
          while (true) {
            const { done, value } = await reader.read()
            if (done) break

            // Decode chunk with stream: true to handle multi-byte characters
            const chunk = decoder.decode(value, { stream: true })
            buffer += chunk

            // Process complete lines
            const lines = buffer.split('\n')
            // Keep the last incomplete line in the buffer
            buffer = lines.pop() || ''

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const dataStr = line.slice(6)
                try {
                  const data = JSON.parse(dataStr)

                  // Capture response_id for conversation continuity
                  // Always use the latest response_id returned by the backend
                  // If LlamaStack decides to start a new conversation (e.g., due to agent/tool mismatch),
                  // it will return a new response_id and we'll use that going forward
                  if (data.response_id) {
                    if (previousResponseId && data.response_id !== previousResponseId) {
                      console.log(`Conversation context changed: ${previousResponseId} -> ${data.response_id}`)
                    }
                    setPreviousResponseId(data.response_id)
                  }

                  if (data.content) {
                    // Update the assistant message by appending new content
                    setMessages(prev => {
                      const newMessages = [...prev]
                      const lastMsg = newMessages[newMessages.length - 1]
                      if (lastMsg && lastMsg.role === 'assistant' && lastMsg.id === assistantMessageId) {
                        newMessages[newMessages.length - 1] = {
                          ...lastMsg,
                          content: lastMsg.content + data.content
                        }
                      }
                      return newMessages
                    })
                  }

                  if (data.reasoning) {
                    // Update the assistant message by appending reasoning text
                    // Use flushSync to force immediate synchronous rendering
                    flushSync(() => {
                      setMessages(prev => {
                        const newMessages = [...prev]
                        const lastMsg = newMessages[newMessages.length - 1]
                        if (lastMsg && lastMsg.role === 'assistant' && lastMsg.id === assistantMessageId) {
                          newMessages[newMessages.length - 1] = {
                            ...lastMsg,
                            reasoning: (lastMsg.reasoning || '') + data.reasoning
                          }
                        }
                        return newMessages
                      })
                    })
                  }

                  if (data.sources) {
                    // Add sources to the assistant message
                    setMessages(prev => {
                      const newMessages = [...prev]
                      const lastMsg = newMessages[newMessages.length - 1]
                      if (lastMsg && lastMsg.role === 'assistant' && lastMsg.id === assistantMessageId) {
                        newMessages[newMessages.length - 1] = {
                          ...lastMsg,
                          sources: data.sources
                        }
                      }
                      return newMessages
                    })
                  }

                  if (data.tool_call) {
                    // Update tool call in the assistant message
                    setMessages(prev => {
                      const newMessages = [...prev]
                      const lastMsg = newMessages[newMessages.length - 1]
                      if (lastMsg && lastMsg.role === 'assistant' && lastMsg.id === assistantMessageId) {
                        const toolCalls = lastMsg.toolCalls || {}
                        newMessages[newMessages.length - 1] = {
                          ...lastMsg,
                          toolCalls: {
                            ...toolCalls,
                            [data.tool_call.id]: {
                              ...toolCalls[data.tool_call.id],
                              ...data.tool_call
                            }
                          }
                        }
                      }
                      return newMessages
                    })
                  }

                  if (data.error) {
                    console.error('Error from backend:', data.error)
                    setMessages(prev => {
                      const newMessages = [...prev]
                      const lastMsg = newMessages[newMessages.length - 1]
                      if (lastMsg && lastMsg.role === 'assistant' && lastMsg.id === assistantMessageId) {
                        newMessages[newMessages.length - 1] = {
                          ...lastMsg,
                          content: lastMsg.content + `\n\nError: ${data.error}`
                        }
                      }
                      return newMessages
                    })
                  }
                } catch (e) {
                  // Ignore JSON parse errors
                }
              }
            }
          }
        }

        // Mark streaming as complete and clean up file references
        setMessages(prev => {
          const newMessages = [...prev]
          const lastMsg = newMessages[newMessages.length - 1]
          if (lastMsg && lastMsg.id === assistantMessageId) {
            // Remove file reference markers like <|file-xxx|>
            const cleanedContent = lastMsg.content.replace(/<\|file-[a-f0-9]+\|>/g, '')

            newMessages[newMessages.length - 1] = {
              ...lastMsg,
              content: cleanedContent,
              isStreaming: false
            }
          }
          return newMessages
        })

      } catch (error) {
        console.error('Error:', error)
        setMessages(prev => {
          const newMessages = [...prev]
          const lastMsg = newMessages[newMessages.length - 1]
          if (lastMsg && lastMsg.id === assistantMessageId) {
            // Clean up any file references before setting error message
            const cleanedContent = lastMsg.content.replace(/<\|file-[a-f0-9]+\|>/g, '')

            newMessages[newMessages.length - 1] = {
              ...lastMsg,
              content: cleanedContent || 'Sorry, I encountered an error. Please try again.',
              isStreaming: false
            }
          }
          return newMessages
        })
      } finally {
        setIsStreaming(false)
        setStreamingMessageId(null)
      }
    },
    [isStreaming, messages, selectedAgent, backendUrl, previousResponseId]
  )

  const handleSubmit: FormEventHandler<HTMLFormElement> = useCallback(
    async (event) => {
      event.preventDefault()
      await submitMessage(inputValue)
      setInputValue('')
    },
    [inputValue, submitMessage]
  )

  const handleReset = useCallback(() => {
    const newMessages: ChatMessage[] = [
      {
        id: nanoid(),
        content: "Hello! I'm your workshop AI assistant. I can help you with questions about the workshop content, troubleshooting, and guidance. What would you like to know?",
        role: 'assistant' as const,
        timestamp: new Date()
      }
    ]
    setMessages(newMessages)
    setInputValue('')
    setIsStreaming(false)
    setStreamingMessageId(null)
    setPreviousResponseId(null)  // Reset conversation continuity

    // Clear localStorage
    if (typeof window !== 'undefined') {
      localStorage.removeItem(STORAGE_KEY_MESSAGES)
    }
  }, [])

  const handleExampleQuestionClick = useCallback((question: string) => {
    if (isStreaming) return
    submitMessage(question)
  }, [isStreaming, submitMessage])

  // Find selected agent name for display
  const selectedAgentName = selectedAgent === 'auto'
    ? 'Auto-select'
    : agents.find(a => a.id === selectedAgent)?.name || 'Unknown'

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-background">
      {/* Header */}
      <div className="flex items-center justify-between border-b bg-muted/50 px-4 py-3 flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="size-2 rounded-full bg-green-500" />
            <span className="font-medium text-sm">Workshop AI Assistant</span>
          </div>
          <div className="h-4 w-px bg-border" />
          <span className="text-muted-foreground text-xs">
            {selectedAgentName}
          </span>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={handleReset}
          className="h-8 px-2"
        >
          <RotateCcwIcon className="size-4" />
          <span className="ml-1">Reset</span>
        </Button>
      </div>

      {/* Conversation Area */}
      <Conversation className="flex-1 min-h-0">
        <ConversationContent className="space-y-4">
          {messages.map((message, messageIdx) => (
            <div key={message.id} className="space-y-3">
              <Message from={message.role}>
                <MessageContent>
                  {message.isStreaming && message.content === '' && !message.reasoning ? (
                    <div className="flex items-center gap-2">
                      <Loader size={14} />
                      <span className="text-muted-foreground text-sm">Thinking...</span>
                    </div>
                  ) : (
                    <>
                      {/* Show reasoning for assistant messages if available */}
                      {message.role === 'assistant' && message.reasoning !== undefined && (
                        <Reasoning isStreaming={message.isStreaming || false}>
                          <ReasoningTrigger />
                          <ReasoningContent>
                            <Response className="grid gap-2 italic text-muted-foreground">
                              {message.reasoning}
                            </Response>

                            {/* Show tool calls inside reasoning - they collapse together */}
                            {message.toolCalls && Object.keys(message.toolCalls).length > 0 && (
                              <div className="mt-4 space-y-2">
                                {Object.values(message.toolCalls).map((toolCall) => (
                                  <Tool key={toolCall.id}>
                                    <ToolHeader type={toolCall.name} state={toolCall.state} />
                                    <ToolContent>
                                      {toolCall.input && <ToolInput input={toolCall.input} />}
                                      {toolCall.output && <ToolOutput output={toolCall.output} errorText={undefined} />}
                                      {toolCall.errorText && <ToolOutput output={undefined} errorText={toolCall.errorText} />}
                                    </ToolContent>
                                  </Tool>
                                ))}
                              </div>
                            )}
                          </ReasoningContent>
                        </Reasoning>
                      )}

                      <div className={message.reasoning !== undefined ? "mt-6" : ""}>
                        <MarkdownRenderer>{message.content}</MarkdownRenderer>
                      </div>

                      {/* Show example questions below the first message */}
                      {messageIdx === 0 && messages.length === 1 && exampleQuestions.length > 0 && (
                        <div className="mt-4 flex flex-col gap-1">
                          {exampleQuestions.map((question, idx) => {
                            const emoji = idx === 0 ? '🚀' : '📚'
                            return (
                              <button
                                key={idx}
                                onClick={() => handleExampleQuestionClick(question)}
                                disabled={isStreaming}
                                className="text-left text-sm text-blue-600 hover:text-blue-800 hover:underline disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                              >
                                {emoji} {question}
                              </button>
                            )
                          })}
                        </div>
                      )}

                      {message.sources && message.sources.length > 0 && (
                        <Sources className="mt-4">
                          <SourcesTrigger count={message.sources.length} />
                          <SourcesContent>
                            {message.sources.map((source, idx) => (
                              <Source
                                key={idx}
                                href={sourceToUrl(source)}
                                title={source.title}
                              />
                            ))}
                          </SourcesContent>
                        </Sources>
                      )}
                    </>
                  )}
                </MessageContent>
                <MessageAvatar
                  src={message.role === 'user' ? 'https://github.com/shadcn.png' : '/_/img/favicon.ico'}
                  name={message.role === 'user' ? 'User' : 'AI'}
                  className={message.role === 'assistant' ? 'size-8 ring-0' : undefined}
                />
              </Message>
            </div>
          ))}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      {/* Input Area */}
      <div className="border-t p-4 flex-shrink-0">
        <PromptInput onSubmit={handleSubmit}>
          <PromptInputTextarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask me anything about the workshop..."
            disabled={isStreaming}
          />
          <PromptInputToolbar>
            <PromptInputTools>
              <PromptInputModelSelect
                value={selectedAgent}
                onValueChange={setSelectedAgent}
                disabled={isStreaming}
              >
                <PromptInputModelSelectTrigger>
                  <PromptInputModelSelectValue />
                </PromptInputModelSelectTrigger>
                <PromptInputModelSelectContent>
                  <PromptInputModelSelectItem value="auto">
                    Auto-select
                  </PromptInputModelSelectItem>
                  {agents.map((agent) => (
                    <PromptInputModelSelectItem key={agent.id} value={agent.id}>
                      {agent.name}
                    </PromptInputModelSelectItem>
                  ))}
                </PromptInputModelSelectContent>
              </PromptInputModelSelect>
            </PromptInputTools>
            <PromptInputSubmit
              disabled={!inputValue.trim() || isStreaming}
              status={isStreaming ? 'streaming' : 'ready'}
            />
          </PromptInputToolbar>
        </PromptInput>
      </div>
    </div>
  )
}
