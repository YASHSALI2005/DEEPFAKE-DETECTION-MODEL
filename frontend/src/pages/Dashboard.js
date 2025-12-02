import React from 'react';
import { motion } from 'framer-motion';
import TestVideo from '../components/TestVideo';

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-gray-900 py-12">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <p className="text-sm uppercase tracking-widest text-blue-400 mb-3">
            No account required
          </p>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-white">
            Upload a video and get an instant deepfake verdict
          </h1>
          <p className="mt-3 text-lg text-gray-300">
            The detector runs entirely through this page—just select a clip and review the results as soon as the model finishes.
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="max-w-3xl mx-auto"
        >
          <TestVideo />
        </motion.div>
      </div>
    </div>
  );
};

export default Dashboard;