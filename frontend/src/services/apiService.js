const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8001/api/v1';

export const generateUUID = () => {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = (Math.random() * 16) | 0;
        const v = c === 'x' ? r : (r & 0x3) | 0x8;
        return v.toString(16);
    });
};

export const apiService = {
    // Non-streaming query
    async submitQuery(query, options = {}) {
        const correlationId = generateUUID();
        try {
            const response = await fetch(`${API_BASE_URL}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Correlation-ID': correlationId,
                },
                body: JSON.stringify({
                    query,
                    options: {
                        max_precedents: options.max_precedents ?? 10,
                        include_excerpts: options.include_excerpts ?? true,
                        ...options,
                    },
                }),
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.message || `HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Query submission failed:', error);
            throw error;
        }
    },

    // Get chat history
    async getChatHistory(limit = 50, offset = 0) {
        try {
            const response = await fetch(
                `${API_BASE_URL}/chat/history?limit=${limit}&offset=${offset}`,
                {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    mode: 'cors',
                    credentials: 'omit',
                }
            );

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Failed to fetch chat history:', error);
            throw error;
        }
    },

    // Clear chat history
    async clearChatHistory() {
        try {
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                },
                mode: 'cors',
                credentials: 'omit',
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Failed to clear chat history:', error);
            throw error;
        }
    },
};
