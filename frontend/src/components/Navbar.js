import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar = () => {
  const location = useLocation();

  return (
    <nav className="bg-gray-900 border-b border-gray-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <svg className="h-8 w-8 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
              <span className="text-white font-bold text-xl">DeepFake Detector</span>
            </Link>
          </div>
          <div className="flex items-center space-x-6">
            <Link
              to="/"
              className={`text-sm font-medium px-3 py-2 transition-colors ${
                location.pathname === '/' ? 'text-white' : 'text-gray-300 hover:text-white'
              }`}
            >
              Home
            </Link>
            <Link
              to="/dashboard"
              className={`text-sm font-medium px-3 py-2 transition-colors ${
                location.pathname === '/dashboard' ? 'text-white' : 'text-gray-300 hover:text-white'
              }`}
            >
              Detector
            </Link>
            <Link
              to="/dashboard"
              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-full text-sm font-medium transition-all transform hover:scale-105"
            >
              Try Now
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;