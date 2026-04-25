import React from 'react';

const ErrorNotification = ({ errorCode, message, onDismiss }) => {
    const getErrorInfo = (code) => {
        const errorMap = {
            CORPUS_NOT_INDEXED: {
                title: 'Corpus Not Ready',
                icon: 'database',
                bgColor: 'bg-yellow-50',
                borderColor: 'border-yellow-200',
                textColor: 'text-yellow-900',
                action: 'Please ingest your PDF corpus in Settings',
            },
            LLM_UNAVAILABLE: {
                title: 'Service Unavailable',
                icon: 'server',
                bgColor: 'bg-orange-50',
                borderColor: 'border-orange-200',
                textColor: 'text-orange-900',
                action: 'Please try again in a few moments',
            },
            EMPTY_QUERY: {
                title: 'Empty Query',
                icon: 'warning',
                bgColor: 'bg-blue-50',
                borderColor: 'border-blue-200',
                textColor: 'text-blue-900',
                action: 'Please enter a valid query',
            },
            INTERNAL_ERROR: {
                title: 'Something Went Wrong',
                icon: 'error',
                bgColor: 'bg-red-50',
                borderColor: 'border-red-200',
                textColor: 'text-red-900',
                action: 'Contact support if this persists',
            },
        };

        return errorMap[code] || {
            title: 'Error',
            icon: 'error',
            bgColor: 'bg-red-50',
            borderColor: 'border-red-200',
            textColor: 'text-red-900',
            action: 'Please try again',
        };
    };

    const errorInfo = getErrorInfo(errorCode);

    return (
        <div className={`${errorInfo.bgColor} border ${errorInfo.borderColor} rounded-lg p-4 mb-4`}>
            <div className="flex items-start justify-between">
                <div className="flex gap-3 flex-1">
                    <div className={`flex-shrink-0 w-5 h-5 ${errorInfo.textColor} mt-0.5`}>
                        {errorInfo.icon === 'error' && (
                            <svg fill="currentColor" viewBox="0 0 20 20">
                                <path
                                    fillRule="evenodd"
                                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                                    clipRule="evenodd"
                                />
                            </svg>
                        )}
                        {errorInfo.icon === 'warning' && (
                            <svg fill="currentColor" viewBox="0 0 20 20">
                                <path
                                    fillRule="evenodd"
                                    d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                                    clipRule="evenodd"
                                />
                            </svg>
                        )}
                        {(errorInfo.icon === 'server' || errorInfo.icon === 'database') && (
                            <svg fill="currentColor" viewBox="0 0 20 20">
                                <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z" />
                            </svg>
                        )}
                    </div>
                    <div className="flex-1">
                        <h3 className={`font-semibold ${errorInfo.textColor}`}>{errorInfo.title}</h3>
                        <p className={`text-sm mt-1 ${errorInfo.textColor}`}>{message}</p>
                        <p className={`text-xs mt-2 opacity-75 ${errorInfo.textColor}`}>{errorInfo.action}</p>
                    </div>
                </div>
                {onDismiss && (
                    <button
                        onClick={onDismiss}
                        className={`flex-shrink-0 ml-3 ${errorInfo.textColor} hover:opacity-75`}
                    >
                        <svg fill="currentColor" viewBox="0 0 20 20" className="w-5 h-5">
                            <path
                                fillRule="evenodd"
                                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                                clipRule="evenodd"
                            />
                        </svg>
                    </button>
                )}
            </div>
        </div>
    );
};

export default ErrorNotification;
