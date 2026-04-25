# Legal Research Chatbot - Frontend

A modern, real-time legal research interface for querying AI-powered legal precedent analysis with streaming responses and advanced research capabilities.

## 🎯 Project Overview

The **Legal Research Chatbot Frontend** is a production-ready React application that provides lawyers and legal researchers with an intelligent interface to:

- **Search legal precedents** using natural language queries
- **Analyze case law** with AI-powered reasoning and classification
- **View real-time processing** through streaming agent thinking panels
- **Compare supporting vs. adverse cases** with strategic recommendations
- **Maintain chat history** for continuous research sessions

The frontend supports two operational modes:
- **Streaming Mode** (default): Real-time visualization of AI reasoning with thinking panel
- **Non-Streaming Mode**: Traditional REST API with instant full responses

---

## 🏗️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│  Navbar (Navigation)  │  Home (Main Chat)  │  Error Handling   │
└─────────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│ ThinkingPanel      │ PrecedentAnalysis   │ ErrorNotification  │
│ (Agent Steps)      │ (Research Formatting) │ (User Alerts)     │
└─────────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  COMMUNICATION LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│ useWebSocket Hook          │      apiService                    │
│ (WebSocket Management)      │ (REST API Wrapper)               │
└─────────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BACKEND SERVICES                             │
├──────────────────────┬──────────────────────┬──────────────────┤
│ WebSocket Stream     │ REST API Endpoint    │ Chat History     │
│ ws://localhost:8001  │ http://localhost:8001│ Management       │
│ /ws/query            │ /api/v1/query        │ /api/v1/chat    │
└──────────────────────┴──────────────────────┴──────────────────┘
```

---

## 📦 System Components

### 1. **Core UI Components** (`src/components/`)

#### **Home.js** - Main Chat Interface
- **Responsibility**: Central orchestration of the chat UI
- **Key Features**:
  - Message history management with auto-scrolling
  - Dual-mode operation (streaming vs. REST)
  - Real-time state updates for thinking steps
  - Integration with WebSocket and REST services
- **State Management**:
  - `chats`: Array of chat messages (user + assistant)
  - `currentThinkingSteps`: Active reasoning steps during streaming
  - `currentStreamedContent`: Token-by-token response content
  - `useStreaming`: Toggle between streaming/REST modes
  - `currentError`: Error state for user feedback
- **Key Methods**:
  - `handleSubmitStreaming()`: Sends query via WebSocket
  - `handleSubmitNonStreaming()`: Sends query via REST API
  - `fetchAndUpdateChatHistory()`: Loads previous conversations

#### **ThinkingPanel.js** - Agent Reasoning Visualization
- **Responsibility**: Display AI agent's step-by-step thinking process
- **Input Props**: `steps`, `isComplete`
- **Phases Displayed**:
  1. **Thinking** - Initial planning
  2. **LLM Thinking** - Decision making
  3. **Tool Call** - Search execution
  4. **Tool Result** - Documents retrieved
  5. **Synthesizing** - Deduplication
  6. **Classifying** - Query classification
  7. **Query Type** - Final classification result
  8. **Reasoning** - Analysis phase
  9. **Streaming** - Response generation
- **Visual Indicators**:
  - Spinning loader for in-progress steps
  - Numbered step badges
  - Expandable/scrollable container (max-height: 300px)
  - Completion checkmark when done

#### **PrecedentAnalysis.js** - Response Formatting
- **Responsibility**: Format and display research results based on query type
- **Two Rendering Modes**:

  **Mode 1: General Query**
  - Direct answer text
  - Supporting documents with relevance scores
  - Document excerpts and citations

  **Mode 2: Research Query** (Precedent Analysis)
  - **Supporting Precedents** (green): Cases supporting the argument
    - Case name
    - Legal principle
    - Factual alignment
    - Relevant excerpts
  - **Adverse Precedents** (red): Cases opposing the argument
    - Case name
    - Risk description
    - Distinguishing argument
    - Relevant excerpts
  - **Strategy Recommendation** (indigo):
    - Priority arguments list
    - Compensation ranges from comparable cases
    - Risk assessment and mitigation strategies
- **Interactivity**: Expandable precedent cards with details on click

#### **Navbar.js** - Navigation & Controls
- **Components**:
  - Logo and branding
  - Streaming mode toggle (boolean switch)
  - "New Chat" button (clears history)
- **Fixed positioning** for persistent visibility
- **Responsive design** with proper spacing

#### **ErrorNotification.js** - Error Display
- **Responsibility**: User-friendly error messaging
- **Error Types Handled**:
  - `CORPUS_NOT_INDEXED`: Prompt to ingest documents
  - `LLM_UNAVAILABLE`: Service temporarily unavailable
  - `EMPTY_QUERY`: Input validation error
  - `INTERNAL_ERROR`: Generic server error
- **Features**:
  - Color-coded error types (yellow, orange, blue, red)
  - Contextual help text
  - Dismissible notification
  - Icon indicators

### 2. **Communication Layer** (`src/services/` & `src/hooks/`)

#### **useWebSocket.js** - WebSocket Management Hook
- **Purpose**: Encapsulate WebSocket lifecycle and message handling
- **Features**:
  - Connection state management (`isConnected`, `isConnecting`)
  - Message queue for messages sent before connection ready
  - Automatic connection retry logic
  - Graceful error handling
- **Public API**:
  - `connect()`: Establish WebSocket connection
  - `send(data)`: Send message (queues if not connected)
  - `disconnect()`: Close connection
  - `isConnected`: Boolean state
  - `isConnecting`: Boolean state
- **Message Parsing**: Automatic JSON parsing with error handling
- **Lifecycle**: Auto-cleanup on component unmount

#### **apiService.js** - REST API Wrapper
- **Responsibility**: Handle HTTP communication with backend
- **Methods**:
  - `submitQuery(query, options)`: POST /api/v1/query
    - Generates correlation ID for request tracing
    - Supports custom options (max_precedents, include_excerpts)
    - Returns structured response with sources and classification
  - `getChatHistory(limit, offset)`: GET /api/v1/chat/history
    - Paginated retrieval with configurable limit/offset
    - Returns array of messages with metadata
  - `clearChatHistory()`: DELETE /api/v1/chat
    - Clears all chat messages from backend
- **Base URL**: Configurable via `REACT_APP_API_BASE_URL` env var
- **Correlation IDs**: UUID generation for request tracking

### 3. **Data Models & Response Structures**

#### **Chat Message Object**
```javascript
{
  id: string,                    // Unique message identifier
  role: 'user' | 'assistant',   // Message author
  content: string,               // Message text
  created_at: ISO8601,          // Timestamp
  
  // Assistant-specific fields:
  query_type: 'research' | 'general_query',
  response: Object,             // Structured response (see below)
  sources_searched: string[],   // List of searched sources
  agent_steps: Array            // Thinking steps (streaming mode)
}
```

#### **Structured Response (Research Mode)**
```javascript
{
  supporting_precedents: [{
    case_name: string,
    legal_principle: string,
    factual_alignment: string,
    excerpt: string
  }],
  
  adverse_precedents: [{
    case_name: string,
    risk_description: string,
    distinguishing_argument: string,
    excerpt: string
  }],
  
  strategy_recommendation: {
    priority_arguments: string[],
    compensation_range: string,
    risks: string[]
  }
}
```

#### **Structured Response (General Query Mode)**
```javascript
{
  answer: string,
  supporting_documents: [{
    case_name: string,
    file_name: string,
    relevance_score: number,     // 0-1 scale
    excerpt: string
  }]
}
```

---

## 🔄 Data Flow Diagrams

### Streaming Mode Flow
```
User Input
    ▼
Home.handleSubmitStreaming()
    ▼
[WebSocket Connect if needed]
    ▼
Send Query via WebSocket
    ▼
Backend Agent Processing
    ▼
[Multiple WebSocket Messages]:
  thinking → llm_thinking → tool_call → tool_result 
  → synthesizing → classifying → query_type → reasoning 
  → streaming → [stream_chunk...] → completed
    ▼
Home.handleWebSocketMessage() Routes Events
    ▼
setCurrentThinkingSteps([...])        [Display in ThinkingPanel]
setCurrentStreamedContent(prev + txt)  [Display response]
setCurrentMessageMetadata({...})       [Store metadata]
    ▼
fetchAndUpdateChatHistory()            [Fetch from REST]
    ▼
Attach agent_steps to Message
    ▼
Display Complete Message with Thinking History
```

### Non-Streaming Mode Flow
```
User Input
    ▼
Home.handleSubmitNonStreaming()
    ▼
apiService.submitQuery()
    ▼
POST /api/v1/query
    ▼
Backend Processes & Returns Complete Response
    ▼
Response.json():
  {
    correlation_id,
    chat_response,
    query_type,
    response,
    sources_searched
  }
    ▼
Create Assistant Message with Full Data
    ▼
Display Message (formatted by PrecedentAnalysis)
```

### Chat History Loading Flow
```
App Mount
    ▼
useEffect([], [])
    ▼
fetchAndUpdateChatHistory()
    ▼
GET /api/v1/chat/history?limit=50&offset=0
    ▼
Response: { messages: [...] }
    ▼
setChats(messages)
    ▼
Display All Previous Messages
```

---

## 🎯 Key Features & Capabilities

### 1. **Real-Time Streaming** (WebSocket)
- **Live Thinking Visualization**: See AI reasoning as it happens
- **Token-by-Token Streaming**: Responses appear word-by-word
- **Tool Call Transparency**: Watch search execution in real-time
- **9-Phase Reasoning Display**: Complete visibility into agent pipeline

### 2. **Research Mode Intelligence**
- **Supporting Cases**: Green-highlighted precedents with alignment scores
- **Adverse Cases**: Red-highlighted with distinguishing arguments
- **Strategy Recommendations**:
  - Priority arguments ranked by strength
  - Compensation ranges from comparable cases
  - Risk assessment and mitigation strategies

### 3. **Dual-Mode Operation**
- **Toggle Button** in Navbar switches between:
  - Streaming (real-time thinking visualization)
  - REST (instant complete responses)
- **State Persists** during session

### 4. **Chat History Management**
- **Auto-Load** previous conversations on startup
- **Pagination Support** via offset/limit parameters
- **Clear History** button removes all messages
- **Persistent IDs** for message correlation

### 5. **Error Handling & Resilience**
- **Error Mapping**: Backend error codes → User-friendly messages
- **WebSocket Reconnection**: Automatic retry on disconnect
- **Message Queuing**: Prevents message loss during reconnection
- **Network Resilience**: Graceful degradation on failure

### 6. **User Experience**
- **Auto-Scrolling**: Always visible message as conversation grows
- **Keyboard Shortcuts** (via backend integration):
  - Enter: Send message
  - Shift+Enter: New line
- **Visual Feedback**: Loading states, completion indicators
- **Responsive Design**: Works on desktop and tablet

---

## 🛠️ Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Framework** | React | 18.3.1 | UI framework |
| **Routing** | React Router DOM | 6.26.1 | Client-side routing |
| **Styling** | Tailwind CSS | 3.4.10 | Utility-first styling |
| **Markdown** | react-markdown | 10.1.0 | Display formatted responses |
| **Communication** | Native WebSocket | ES6 | Real-time streaming |
| **HTTP Client** | Fetch API | ES6 | REST requests |
| **Testing** | React Testing Library | 13.4.0 | Component testing |
| **Build Tool** | React Scripts | 5.0.1 | CRA build system |

---

## 📁 Project Structure

```
frontend/
├── public/
│   ├── index.html              # HTML entry point
│   ├── manifest.json           # PWA manifest
│   └── robots.txt
│
├── src/
│   ├── components/
│   │   ├── Home.js             # Main chat interface
│   │   ├── ThinkingPanel.js    # Agent reasoning display
│   │   ├── PrecedentAnalysis.js# Research result formatting
│   │   ├── Navbar.js           # Navigation bar
│   │   └── ErrorNotification.js# Error alerts
│   │
│   ├── hooks/
│   │   └── useWebSocket.js     # WebSocket management
│   │
│   ├── services/
│   │   └── apiService.js       # REST API wrapper
│   │
│   ├── App.js                  # Root router component
│   ├── App.css                 # App-level styles
│   ├── index.js                # React DOM render
│   ├── index.css               # Global styles
│   └── setupTests.js           # Test configuration
│
├── package.json
├── tailwind.config.js
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Node.js 14+ 
- npm 6+ or yarn
- Backend running on localhost:8001 (or configured endpoint)

### Installation
```bash
npm install
```

### Development
```bash
npm start
# Opens http://localhost:3000
# Backend must be running on http://localhost:8001
```

### Production Build
```bash
npm run build
# Creates optimized production bundle in `build/` folder
```

### Configuration

#### API Endpoint
Set via environment variable:
```bash
REACT_APP_API_BASE_URL=https://api.yourdomain.com/api/v1 npm start
```

Or edit in `src/services/apiService.js`:
```javascript
const API_BASE_URL = 'https://api.yourdomain.com/api/v1';
```

#### WebSocket Endpoint
Edit in `src/components/Home.js`:
```javascript
// For development
'ws://localhost:8001/ws/query'

// For production (HTTPS)
'wss://api.yourdomain.com/ws/query'
```

---

## 🔌 Backend Integration Points

### WebSocket Messages
The frontend listens for these message types:

| Message Type | Payload | Handler |
|-------------|---------|---------|
| `thinking` | `{ type, message }` | Update thinking steps |
| `llm_thinking` | `{ type, message }` | Update thinking steps |
| `tool_call` | `{ type, tool, args }` | Update thinking steps |
| `tool_result` | `{ type, total_returned, ... }` | Update thinking steps |
| `synthesizing` | `{ type, unique_documents }` | Update thinking steps |
| `classifying` | `{ type }` | Update thinking steps |
| `query_type` | `{ type, is_research }` | Update thinking steps |
| `reasoning` | `{ type, message }` | Update thinking steps |
| `streaming` | `{ type }` | Update thinking steps |
| `stream_chunk` | `{ type, content }` | Append to response |
| `completed` | `{ message_id, query_type, ... }` | Finalize message |
| `error` | `{ error_code, message }` | Display error |

### REST API Endpoints

#### Submit Query (Non-Streaming)
```
POST /api/v1/query
Headers: { 'X-Correlation-ID': uuid }
Body: {
  query: string,
  options: {
    max_precedents: number,
    include_excerpts: boolean
  }
}
Response: {
  correlation_id: string,
  chat_response: string,
  query_type: string,
  response: Object,
  sources_searched: string[]
}
```

#### Get Chat History
```
GET /api/v1/chat/history?limit=50&offset=0
Response: { messages: Array<Message> }
```

#### Clear Chat
```
DELETE /api/v1/chat
Response: { success: boolean }
```

---

## 🧪 Testing

### Run Tests
```bash
npm test
```

### Test Files
- `src/App.test.js` - Router and app-level tests

---

## 📊 Component Dependency Graph

```
App
 └── Home (Main Logic)
      ├── Navbar (Controls & Navigation)
      │   └── Uses: onStreamingChange, onNewChat callbacks
      │
      ├── useWebSocket Hook (Streaming)
      │   └── Manages: WebSocket connection, message queue
      │
      ├── apiService (REST)
      │   └── Methods: submitQuery, getChatHistory, clearChatHistory
      │
      ├── Chat Message Display (mapping)
      │   ├── User Message (aligned right)
      │   ├── ThinkingPanel (if available)
      │   │   └── Displays: step-by-step reasoning
      │   │
      │   ├── PrecedentAnalysis (response formatting)
      │   │   ├── For 'research' mode:
      │   │   │   ├── Supporting Precedents
      │   │   │   ├── Adverse Precedents
      │   │   │   └── Strategy Recommendation
      │   │   │
      │   │   └── For 'general_query' mode:
      │   │       ├── Answer text
      │   │       └── Supporting Documents
      │   │
      │   └── ErrorNotification (if error)
      │       └── Displays: error message & action
      │
      └── Input Area
          ├── Textarea (question input)
          ├── Submit Button
          ├── Mode Toggle (streaming/REST)
          └── Clear History Button
```

---

## 🔐 Security & Best Practices

- **CORS Enabled**: Configured for cross-origin requests
- **UUID Generation**: Client-side correlation ID for request tracking
- **Error Mapping**: Prevents leaking internal error details
- **Input Validation**: Query text trimming before submission
- **WebSocket Auth** (backend-side): Implement token-based auth as needed

---

## 📈 Performance Considerations

1. **Message Virtualization**: For large chat histories, consider virtual scrolling
2. **Lazy Loading**: Think steps initially limited to 300px container
3. **Auto-Scrolling**: Uses smooth behavior for better UX
4. **Streaming**: Token-by-token delivery reduces perceived latency
5. **Error Recovery**: Message queue prevents loss during reconnection

---

## 🚧 Future Enhancements

- [ ] Message editing and deletion
- [ ] Conversation search and filtering
- [ ] Export conversations to PDF
- [ ] User authentication and profiles
- [ ] Conversation branching/forking
- [ ] Advanced export (docx, md, etc.)
- [ ] Thinking panel customization (detail levels)
- [ ] Message reactions/annotations
- [ ] Shared conversations/collaboration

---

## 📝 Available Scripts

In the project directory, you can run:

### `npm start`
Runs the app in development mode at [http://localhost:3000](http://localhost:3000)

### `npm run build`
Builds the app for production to the `build` folder

### `npm test`
Launches the test runner in interactive watch mode

---

## 📚 Documentation

- [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md) - Feature implementation details
- [INTEGRATION_GUIDE.md](./INTEGRATION_GUIDE.md) - Backend integration guide
- [QUICKSTART.md](./QUICKSTART.md) - Quick setup instructions
- [THINKING_PANEL_FLOW.md](./THINKING_PANEL_FLOW.md) - Thinking panel details
- [VERIFICATION_CHECKLIST.md](./VERIFICATION_CHECKLIST.md) - Testing checklist


