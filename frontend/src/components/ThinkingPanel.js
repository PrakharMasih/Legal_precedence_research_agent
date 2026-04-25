import React from 'react';

const ThinkingPanel = ({ steps, isComplete }) => {
    return (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
            <div className="flex items-center mb-3">
                <h3 className="text-sm font-semibold text-blue-900 flex items-center gap-2">
                    {!isComplete && (
                        <svg
                            className="w-4 h-4 text-blue-600 animate-spin"
                            fill="none"
                            viewBox="0 0 24 24"
                        >
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
                    )}
                    Thinking Process
                </h3>
            </div>

            <div className="space-y-2 max-h-[300px] overflow-y-auto">
                {steps.map((step, idx) => (
                    <div key={idx} className="flex gap-3 text-xs">
                        <div className="flex-shrink-0 w-6 h-6 bg-blue-200 rounded-full flex items-center justify-center text-blue-700 font-semibold">
                            {step.step || idx + 1}
                        </div>
                        <div className="flex-grow pt-1">
                            <div className="text-blue-900">
                                {(() => {
                                    switch (step.type) {
                                        case 'thinking':
                                            return `Thinking: ${step.message}`;
                                        case 'llm_thinking':
                                            return `LLM Planning: ${step.message}`;
                                        case 'tool_call':
                                            return `Tool Call: ${step.tool} with query "${step.args?.query}"`;
                                        case 'tool_result':
                                            return `Tool Result: Found ${step.total_returned} documents`;
                                        case 'synthesizing':
                                            return `Synthesizing: ${step.unique_documents?.length || 0} unique documents`;
                                        case 'query_type':
                                            return `Query Type: ${step.is_research ? 'Precedent Research' : 'General Query'}`;
                                        case 'reasoning':
                                            return `Reasoning: ${step.message}`;
                                        case 'streaming':
                                            return `Streaming response...`;
                                        default:
                                            return step.message || `Step ${step.step}`;
                                    }
                                })()}
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {isComplete && (
                <div className="mt-3 text-xs text-blue-700 flex items-center gap-2">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path
                            fillRule="evenodd"
                            d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                            clipRule="evenodd"
                        />
                    </svg>
                    Analysis complete
                </div>
            )}
        </div>
    );
};

export default ThinkingPanel;
