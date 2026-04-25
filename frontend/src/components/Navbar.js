import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = ({ useStreaming, onStreamingChange, onNewChat }) => {
  return (
    <div className="fixed top-0 w-full z-50 bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo and Title */}
          <Link to="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
              <svg
                className="w-5 h-5 text-white"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
                <path fillRule="evenodd" d="M4 5a2 2 0 012-2 1 1 0 100-2H6a6 6 0 100 12H4a2 2 0 01-2-2v-4a2 2 0 012-2zm15-3a1 1 0 11-2 0 1 1 0 012 0zm-4-3a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" />
              </svg>
            </div>
            <span className="font-bold text-xl text-gray-900">Legal Researcher</span>
          </Link>

          {/* Controls */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <label className="text-xs text-gray-600 font-medium">Real-time:</label>
              <button
                onClick={() => onStreamingChange(!useStreaming)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${useStreaming ? 'bg-indigo-600' : 'bg-gray-300'}`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${useStreaming ? 'translate-x-6' : 'translate-x-1'}`}
                />
              </button>
            </div>
            <button
              onClick={onNewChat}
              className="text-xs px-3 py-1.5 bg-red-100 text-red-700 rounded hover:bg-red-200 font-medium transition-colors"
            >
              New Chat
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Navbar;
