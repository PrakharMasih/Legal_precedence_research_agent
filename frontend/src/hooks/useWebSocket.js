import { useEffect, useRef, useCallback, useState } from 'react';

export const useWebSocket = (url, onMessage, onError) => {
    const wsRef = useRef(null);
    const messageQueueRef = useRef([]);
    const [isConnected, setIsConnected] = useState(false);
    const [isConnecting, setIsConnecting] = useState(false);

    const connect = useCallback(() => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            return;
        }

        setIsConnecting(true);
        try {
            const ws = new WebSocket(url);

            ws.onopen = () => {
                console.log('WebSocket connected');
                setIsConnected(true);
                setIsConnecting(false);

                // Flush queued messages
                while (messageQueueRef.current.length > 0) {
                    const queuedMessage = messageQueueRef.current.shift();
                    ws.send(JSON.stringify(queuedMessage));
                }
            };

            ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    onMessage(msg);
                } catch (err) {
                    console.error('Failed to parse WebSocket message:', err);
                    onError?.(err);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                onError?.(error);
                setIsConnected(false);
                setIsConnecting(false);
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected');
                setIsConnected(false);
                setIsConnecting(false);
            };

            wsRef.current = ws;
        } catch (err) {
            console.error('Failed to create WebSocket:', err);
            onError?.(err);
            setIsConnecting(false);
        }
    }, [url, onMessage, onError]);

    const send = useCallback((data) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(data));
        } else {
            // Queue message if not connected yet
            messageQueueRef.current.push(data);
            console.log('Message queued, waiting for connection...');
        }
    }, []);

    const disconnect = useCallback(() => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
            setIsConnected(false);
        }
    }, []);

    useEffect(() => {
        return () => {
            disconnect();
        };
    }, [disconnect]);

    return { connect, send, disconnect, isConnected, isConnecting };
};
