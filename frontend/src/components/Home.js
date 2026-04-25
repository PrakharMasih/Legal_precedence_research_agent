import React, { useEffect, useState, useCallback, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { useWebSocket } from '../hooks/useWebSocket';
import { apiService } from '../services/apiService';
import Navbar from './Navbar';
import ThinkingPanel from './ThinkingPanel';
import PrecedentAnalysis from './PrecedentAnalysis';
import ErrorNotification from './ErrorNotification';

const Home = () => {
  const [chats, setChats] = useState([]);
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentThinkingSteps, setCurrentThinkingSteps] = useState([]);
  const [currentStreamedContent, setCurrentStreamedContent] = useState('');
  const [currentError, setCurrentError] = useState(null);
  const [currentMessageMetadata, setCurrentMessageMetadata] = useState(null);
  const [expandedThinking, setExpandedThinking] = useState({});
  const [expandCurrentThinking, setExpandCurrentThinking] = useState(true);
  const messagesEndRef = useRef(null);
  const [useStreaming, setUseStreaming] = useState(true);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chats, currentStreamedContent, currentThinkingSteps]);

  // WebSocket handlers
  const handleWebSocketMessage = useCallback((msg) => {
    switch (msg.type) {
      case 'thinking':
      case 'llm_thinking':
      case 'synthesizing':
      case 'classifying':
      case 'query_type':
      case 'reasoning':
      case 'streaming':
        setCurrentThinkingSteps((prev) => [...prev, msg]);
        setCurrentError(null);
        break;

      case 'tool_call':
      case 'tool_result':
        setCurrentThinkingSteps((prev) => [...prev, msg]);
        break;

      case 'stream_chunk':
        setCurrentStreamedContent((prev) => prev + msg.content);
        break;

      case 'completed':
        setCurrentMessageMetadata({
          message_id: msg.message_id,
          query_type: msg.query_type,
          sources_searched: msg.sources_searched,
          processing_time_ms: msg.processing_time_ms,
        });
        
        // Update thinking steps with completion marker
        setCurrentThinkingSteps((prev) => {
          const completedSteps = [...prev, { type: 'completed', message_id: msg.message_id }];
          console.log('Thinking steps completed:', completedSteps.length, 'steps');
          
          // Store reference to completed steps for use in async callback
          const stepsToAttach = completedSteps;
          
          // Fetch full history to get structured response and attach thinking steps
          (async () => {
            try {
              const updatedMessages = await fetchAndUpdateChatHistory();
              console.log('Fetched chat history:', updatedMessages.length, 'messages');
              
              // Find the assistant message that was just completed
              const assistantMessage = updatedMessages.find(
                (m) => m.id === msg.message_id || m.message_id === msg.message_id
              );
              
              console.log('Looking for message with id:', msg.message_id);
              console.log('Found assistant message:', !!assistantMessage, assistantMessage?.id || assistantMessage?.message_id);
              
              // Attach thinking steps to the message
              if (assistantMessage) {
                setChats((prevChats) => {
                  const updated = prevChats.map((c) => {
                    if (c.id === assistantMessage.id || c.message_id === assistantMessage.id) {
                      console.log('Attaching', stepsToAttach.length, 'steps to message:', c.id);
                      return { ...c, agent_steps: stepsToAttach };
                    }
                    return c;
                  });
                  console.log('Updated chats, checking agent_steps:', updated.find(m => m.agent_steps)?.agent_steps?.length);
                  return updated;
                });
              } else {
                console.warn('No assistant message found to attach thinking steps');
              }
            } finally {
              // Stop loading - message with thinking steps should now be visible
              console.log('Completed message handling, setting isLoading to false');
              setIsLoading(false);
            }
          })();
          
          return completedSteps;
        });
        break;

      case 'error':
        setCurrentError({
          code: msg.error_code,
          message: msg.message,
        });
        setIsLoading(false);
        break;

      default:
        console.log('Unknown message type:', msg.type);
    }
  }, []);

  const handleWebSocketError = useCallback((error) => {
    console.error('WebSocket error:', error);
    setCurrentError({
      code: 'INTERNAL_ERROR',
      message: 'Connection error. Please try again.',
    });
    setIsLoading(false);
  }, []);

  const { connect, send, isConnected, isConnecting } = useWebSocket(
    'ws://localhost:8001/ws/query',
    handleWebSocketMessage,
    handleWebSocketError
  );

  // Load chat history on mount
  useEffect(() => {
    fetchAndUpdateChatHistory();
  }, []);

  const fetchAndUpdateChatHistory = async () => {
    try {
      const data = await apiService.getChatHistory(50, 0);
      setChats(data.messages ?? []);
      return data.messages ?? [];
    } catch (error) {
      console.error('Failed to fetch conversations:', error);
      return [];
    }
  };

  const handleSubmitStreaming = (queryText) => {
    if (!isConnected && !isConnecting) {
      connect();
    }

    // Reset streaming state
    setCurrentThinkingSteps([]);
    setCurrentStreamedContent('');
    setCurrentMessageMetadata(null);
    setCurrentError(null);

    // Add user message immediately
    setChats((prev) => [
      ...prev,
      {
        id: `temp-${Date.now()}`,
        role: 'user',
        content: queryText,
        created_at: new Date().toISOString(),
      },
    ]);

    // Send query
    send({
      query: queryText,
      mode: 'auto',
      options: {
        max_precedents: 10,
        include_excerpts: true,
      },
    });
  };

  const handleSubmitNonStreaming = async (queryText) => {
    // Add user message first
    setChats((prev) => [
      ...prev,
      {
        id: `user-${Date.now()}`,
        role: 'user',
        content: queryText,
        created_at: new Date().toISOString(),
      },
    ]);

    setCurrentError(null);

    try {
      const result = await apiService.submitQuery(queryText);

      // Add assistant message after processing is complete
      setChats((prev) => [
        ...prev,
        {
          id: result.correlation_id,
          role: 'assistant',
          content: result.chat_response,
          query_type: result.query_type,
          response: result.response,
          sources_searched: result.sources_searched,
          created_at: new Date().toISOString(),
        },
      ]);
    } catch (error) {
      setCurrentError({
        code: 'INTERNAL_ERROR',
        message: error.message || 'Failed to submit query',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = () => {
    if (!question.trim()) return;
    setIsLoading(true);

    if (useStreaming) {
      handleSubmitStreaming(question);
    } else {
      handleSubmitNonStreaming(question);
    }

    setQuestion('');
  };

  const handleClearHistory = async () => {
    try {
      await apiService.clearChatHistory();
      setChats([]);
      setCurrentThinkingSteps([]);
      setCurrentStreamedContent('');
    } catch (error) {
      console.error('Failed to clear history:', error);
    }
  };

  return (
    <>
      <Navbar
        useStreaming={useStreaming}
        onStreamingChange={setUseStreaming}
        onNewChat={handleClearHistory}
      />
      <div className="flex flex-col h-screen bg-gray-50 pt-16">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {chats.length === 0 && currentThinkingSteps.length === 0 && (
            <div className="flex items-center justify-center h-full">
              <div className="bg-indigo-100 p-8 rounded-lg text-center max-w-md">
                <svg className="w-12 h-12 text-indigo-600 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-5.36 4.364l-.707.707M5.05 7.05a7 7 0 1 0 9.9 9.9m-9.9-9.9l.707.707" />
                </svg>
                <p className="text-indigo-900 font-semibold">Welcome to Legal Research</p>
                <p className="text-indigo-700 text-sm mt-2">
                  Ask any legal question and I'll search through the knowledge base to provide research-backed answers with supporting precedents.
                </p>
              </div>
            </div>
          )}

          {/* Chat history */}
          {chats.map((msg, index) => {
            const thinkingKey = `${msg.id}-${index}`;
            const isThinkingExpanded = expandedThinking[thinkingKey] !== false;
            const assistantMsg = msg.role === 'assistant' ? msg : chats[index + 1];
            const userMsg = msg.role === 'user' ? msg : null;

            // Only render pairs of user + assistant messages
            if (msg.role !== 'user') return null;

            const hasThinking = assistantMsg?.agent_steps && assistantMsg.agent_steps.length > 0;

            return (
              <div key={msg.id || index} className="mb-4">
                {/* User Message */}
                <div className="flex justify-end mb-2">
                  <div className="max-w-[75%] bg-indigo-600 text-white p-4 rounded-lg shadow-sm">
                    <p className="text-sm">{msg.content}</p>
                  </div>
                </div>

                {/* Thinking Panel - Between User and Assistant */}
                {hasThinking && (
                  <div className="flex justify-start mb-3 max-w-[85%]">
                    <div className="w-full">
                      <button
                        onClick={() => setExpandedThinking(prev => ({ ...prev, [thinkingKey]: !isThinkingExpanded }))}
                        className="text-xs font-semibold text-blue-600 hover:text-blue-800 flex items-center gap-1 mb-2"
                      >
                        {isThinkingExpanded ? (
                          <>
                            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                            </svg>
                            Hide Thinking
                          </>
                        ) : (
                          <>
                            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                            </svg>
                            Show Thinking
                          </>
                        )}
                      </button>
                      {isThinkingExpanded && (
                        <ThinkingPanel steps={assistantMsg.agent_steps} isComplete={true} />
                      )}
                    </div>
                  </div>
                )}

                {/* Assistant Message */}
                {assistantMsg && (
                  <div className="flex justify-start mb-4">
                    <div className="max-w-[85%] space-y-3 w-full">
                      {/* Chat response */}
                      <div className="bg-white border border-gray-200 text-gray-900 p-4 rounded-lg shadow-sm">
                        <ReactMarkdown
                          components={{
                            h1: ({ node, ...props }) => <h1 className="text-2xl font-bold mb-3 mt-4 text-gray-900" {...props} />,
                            h2: ({ node, ...props }) => <h2 className="text-xl font-bold mb-2 mt-3 text-gray-900" {...props} />,
                            h3: ({ node, ...props }) => <h3 className="text-lg font-semibold mb-2 mt-2 text-gray-800" {...props} />,
                            h4: ({ node, ...props }) => <h4 className="text-base font-semibold mb-1 mt-2 text-gray-800" {...props} />,
                            p: ({ node, ...props }) => <p className="mb-2 text-gray-700" {...props} />,
                            ul: ({ node, ...props }) => <ul className="list-disc list-inside mb-2 space-y-1 text-gray-700" {...props} />,
                            ol: ({ node, ...props }) => <ol className="list-decimal list-inside mb-2 space-y-1 text-gray-700" {...props} />,
                            li: ({ node, ...props }) => <li className="text-gray-700" {...props} />,
                            strong: ({ node, ...props }) => <strong className="font-bold text-gray-900" {...props} />,
                            em: ({ node, ...props }) => <em className="italic text-gray-700" {...props} />,
                            code: ({ node, inline, ...props }) =>
                              inline ?
                                <code className="bg-gray-100 px-2 py-1 rounded text-red-600 font-mono text-sm" {...props} /> :
                                <code className="bg-gray-100 p-2 rounded block overflow-x-auto text-sm font-mono mb-2" {...props} />,
                            blockquote: ({ node, ...props }) => <blockquote className="border-l-4 border-indigo-500 pl-4 italic text-gray-600 my-2" {...props} />,
                            a: ({ node, ...props }) => <a className="text-indigo-600 hover:text-indigo-800 underline" {...props} />,
                            hr: ({ node, ...props }) => <hr className="my-4 border-gray-300" {...props} />,
                          }}
                        >
                          {assistantMsg.content}
                        </ReactMarkdown>
                      </div>

                      {/* Structured response */}
                      {(assistantMsg.response || assistantMsg.raw_response) && (
                        <div className="mt-2">
                          <PrecedentAnalysis response={assistantMsg.response || assistantMsg.raw_response} queryType={assistantMsg.query_type} />
                        </div>
                      )}

                      {/* Metadata */}
                      {(assistantMsg.sources_searched || assistantMsg.query_type) && (
                        <div className="text-xs text-gray-500 px-1 py-2">
                          <span className="inline-block bg-gray-100 px-2 py-1 rounded mr-2">
                            Type: {assistantMsg.query_type || 'unknown'}
                          </span>
                          {assistantMsg.sources_searched && (
                            <span className="inline-block bg-gray-100 px-2 py-1 rounded">
                              Sources: {assistantMsg.sources_searched}
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          {/* Current streaming response */}
          {isLoading && currentThinkingSteps.length > 0 && (
            <div>
              {/* Expandable thinking panel */}
              <div className="flex justify-start mb-3 max-w-[85%]">
                <div className="w-full">
                  <button
                    onClick={() => setExpandCurrentThinking(!expandCurrentThinking)}
                    className="text-xs font-semibold text-blue-600 hover:text-blue-800 flex items-center gap-1 mb-2"
                  >
                    {expandCurrentThinking ? (
                      <>
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                        </svg>
                        Hide Thinking
                      </>
                    ) : (
                      <>
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                        </svg>
                        Show Thinking
                      </>
                    )}
                  </button>
                  {expandCurrentThinking && (
                    <ThinkingPanel steps={currentThinkingSteps} isComplete={false} />
                  )}
                </div>
              </div>

              {/* Streamed content */}
              <div className="flex justify-start">
                <div className="max-w-[85%] space-y-3 w-full">
                  {currentStreamedContent && (
                    <div className="bg-white border border-gray-200 text-gray-900 p-4 rounded-lg shadow-sm">
                      <ReactMarkdown
                        components={{
                          h1: ({ node, ...props }) => <h1 className="text-2xl font-bold mb-3 mt-4 text-gray-900" {...props} />,
                          h2: ({ node, ...props }) => <h2 className="text-xl font-bold mb-2 mt-3 text-gray-900" {...props} />,
                          h3: ({ node, ...props }) => <h3 className="text-lg font-semibold mb-2 mt-2 text-gray-800" {...props} />,
                          h4: ({ node, ...props }) => <h4 className="text-base font-semibold mb-1 mt-2 text-gray-800" {...props} />,
                          p: ({ node, ...props }) => <p className="mb-2 text-gray-700" {...props} />,
                          ul: ({ node, ...props }) => <ul className="list-disc list-inside mb-2 space-y-1 text-gray-700" {...props} />,
                          ol: ({ node, ...props }) => <ol className="list-decimal list-inside mb-2 space-y-1 text-gray-700" {...props} />,
                          li: ({ node, ...props }) => <li className="text-gray-700" {...props} />,
                          strong: ({ node, ...props }) => <strong className="font-bold text-gray-900" {...props} />,
                          em: ({ node, ...props }) => <em className="italic text-gray-700" {...props} />,
                          code: ({ node, inline, ...props }) =>
                            inline ?
                              <code className="bg-gray-100 px-2 py-1 rounded text-red-600 font-mono text-sm" {...props} /> :
                              <code className="bg-gray-100 p-2 rounded block overflow-x-auto text-sm font-mono mb-2" {...props} />,
                          blockquote: ({ node, ...props }) => <blockquote className="border-l-4 border-indigo-500 pl-4 italic text-gray-600 my-2" {...props} />,
                          a: ({ node, ...props }) => <a className="text-indigo-600 hover:text-indigo-800 underline" {...props} />,
                          hr: ({ node, ...props }) => <hr className="my-4 border-gray-300" {...props} />,
                        }}
                      >
                        {currentStreamedContent}
                      </ReactMarkdown>
                    </div>
                  )}

                  {/* Metadata preview */}
                  {currentMessageMetadata && (
                    <div className="text-xs text-gray-500 px-1 py-2">
                      <span className="inline-block bg-gray-100 px-2 py-1 rounded mr-2">
                        Type: {currentMessageMetadata.query_type}
                      </span>
                      <span className="inline-block bg-gray-100 px-2 py-1 rounded">
                        Sources: {currentMessageMetadata.sources_searched}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Error notification */}
          {currentError && (
            <div className="max-w-[85%]">
              <ErrorNotification
                errorCode={currentError.code}
                message={currentError.message}
                onDismiss={() => setCurrentError(null)}
              />
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-white border-t border-gray-200 p-4 sticky bottom-0 shadow-md">
          <div className="flex gap-2 max-w-4xl mx-auto">
            <input
              type="text"
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-sm"
              placeholder="Ask your legal question..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit();
                }
              }}
              disabled={isLoading}
            />
            <button
              onClick={handleSubmit}
              disabled={isLoading || !question.trim()}
              className={`px-6 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-all font-medium text-sm ${isLoading || !question.trim() ? 'opacity-50 cursor-not-allowed' : ''
                }`}
            >
              {isLoading ? (
                <span className="flex items-center gap-2">
                  <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                </span>
              ) : (
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5.951-1.488 5.951 1.488a1 1 0 001.169-1.409l-7-14z" />
                </svg>
              )}
            </button>
          </div>
        </div>
      </div>
    </>
  );
};

export default Home;