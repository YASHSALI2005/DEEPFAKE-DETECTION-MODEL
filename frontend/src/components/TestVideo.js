import React, { useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';

const TestVideo = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [verdict, setVerdict] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError(null);
    setVerdict(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a video file');
      return;
    }

    setLoading(true);
    setError(null);
    setVerdict(null);
    setResults(null);

    const formData = new FormData();
    formData.append('video', file);

    try {
      console.log('Sending video for analysis...');
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/api/videos/test`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        withCredentials: true
      });

      console.log('Server response:', response.data);

      if (response.data.error) {
        setError(response.data.error);
        return;
      }

      const results = response.data.results;
      console.log('Processing results:', results);
      
      if (!results) {
        setError('No results received from the server');
        return;
      }

      // Set verdict first
      setVerdict({
        label: results.verdict,
        score: results.score
      });
      
      // Then set results with sample faces
      if (results.sampleFaces && results.sampleFaces.length > 0) {
        console.log(`Received ${results.sampleFaces.length} sample faces`);
        setResults({
          sampleFaces: results.sampleFaces,
          predictions: results.predictions || []
        });
      } else {
        console.log('No sample faces received from server');
      }
      
      console.log('Verdict set:', results.verdict, 'Score:', results.score);
    } catch (err) {
      console.error('Error details:', err);
      setError(err.response?.data?.error || 'Error processing video. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
      <h2 className="text-2xl font-bold text-white mb-4">Test Video</h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Select Video File
          </label>
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-300
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-500 file:text-white
              hover:file:bg-blue-600"
          />
          {file && (
            <p className="mt-2 text-sm text-gray-400">
              Selected: {file.name}
            </p>
          )}
        </div>

        {error && (
          <div className="text-red-500 text-sm bg-red-900/20 p-3 rounded-md">
            {error}
            {error.includes('backend server') && (
              <p className="mt-2 text-xs">
                Please ensure that:
                <br />1. The backend server is running (cd backend && npm start)
                <br />2. The .env file has the correct API URL
                <br />3. The uploads directory exists in the backend
              </p>
            )}
          </div>
        )}

        <button
          type="submit"
          disabled={loading || !file}
          className={`w-full py-2 px-4 rounded-md text-white font-medium
            ${loading || !file
              ? 'bg-gray-600 cursor-not-allowed'
              : 'bg-blue-500 hover:bg-blue-600'
            }`}
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Testing Video...
            </span>
          ) : 'Test Video'}
        </button>
      </form>

      {verdict && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-4 rounded-lg text-center"
        >
          <div 
            className={`inline-block px-8 py-4 rounded-lg ${
              verdict.label === 'FAKE' 
                ? 'bg-red-600' 
                : 'bg-green-600'
            }`}
            style={{
              boxShadow: '0 0 10px rgba(0,0,0,0.5)'
            }}
          >
            <h3 
              className="text-6xl font-black text-white"
              style={{
                textShadow: '2px 2px 4px rgba(0,0,0,0.6)',
                letterSpacing: '0.05em',
                WebkitTextStroke: '1px black'
              }}
            >
              {verdict.label}
            </h3>
          </div>

          {/* Display Performance Metrics */}
          {results && results.metrics && (
            <div className="mt-6 grid grid-cols-2 gap-4">
              <div className="bg-gray-700 p-4 rounded-lg">
                <h4 className="text-lg font-medium text-white mb-2">Precision</h4>
                <p className="text-2xl font-bold text-blue-400">{results.metrics.precision.toFixed(2)}%</p>
              </div>
              <div className="bg-gray-700 p-4 rounded-lg">
                <h4 className="text-lg font-medium text-white mb-2">Recall</h4>
                <p className="text-2xl font-bold text-green-400">{results.metrics.recall.toFixed(2)}%</p>
              </div>
              <div className="bg-gray-700 p-4 rounded-lg">
                <h4 className="text-lg font-medium text-white mb-2">F1 Score</h4>
                <p className="text-2xl font-bold text-purple-400">{results.metrics.f1_score.toFixed(2)}%</p>
              </div>
              <div className="bg-gray-700 p-4 rounded-lg">
                <h4 className="text-lg font-medium text-white mb-2">Accuracy</h4>
                <p className="text-2xl font-bold text-yellow-400">{results.metrics.accuracy.toFixed(2)}%</p>
              </div>
            </div>
          )}
        </motion.div>
      )}

      {results && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6"
        >
          <h3 className="text-xl font-semibold text-white mb-4">Analysis Results</h3>
          
          {/* Display sample faces */}
          {results.sampleFaces && results.sampleFaces.length > 0 && (
            <div className="mb-6">
              <h4 className="text-lg font-medium text-white mb-2">Sample Faces from Video</h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                {results.sampleFaces.map((face, index) => (
                  <div key={index} className="relative">
                    <img
                      src={`data:image/jpeg;base64,${face}`}
                      alt={`Face ${index + 1}`}
                      className="w-full h-48 object-cover rounded-lg"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Display prediction plot */}
          {results.plot && (
            <div className="mb-6">
              <h4 className="text-lg font-medium text-white mb-2">Prediction Plot</h4>
              <img
                src={`data:image/png;base64,${results.plot}`}
                alt="Prediction Plot"
                className="w-full rounded-lg"
              />
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
};

export default TestVideo; 