import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

const Home = () => {
  return (
    <div className="bg-gray-900 min-h-screen">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-900/20 to-purple-900/20"></div>
        <div className="max-w-7xl mx-auto">
          <div className="relative z-10 pb-8 sm:pb-16 md:pb-20 lg:pb-28 xl:pb-32">
            <main className="mt-10 mx-auto max-w-7xl px-4 sm:mt-12 sm:px-6 md:mt-16 lg:mt-20 lg:px-8 xl:mt-28">
              <div className="lg:grid lg:grid-cols-2 lg:gap-8 lg:items-center">
                {/* Left Content */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                  className="sm:text-center lg:text-left"
                >
                  <h1 className="text-4xl tracking-tight font-extrabold text-white sm:text-5xl md:text-6xl">
                    <span className="block">Be Protected Against</span>
                    <span className="block text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500">
                      Deepfakes!
                    </span>
                  </h1>
                  <p className="mt-3 text-base text-gray-300 sm:mt-5 sm:text-lg sm:max-w-xl sm:mx-auto md:mt-5 md:text-xl lg:mx-0">
                    We offer an AI tool that can identify if an audio or video is a deepfake or real.
                  </p>
                  <div className="mt-5 sm:mt-8 sm:flex sm:justify-center lg:justify-start">
                    <div className="rounded-md shadow">
                      <Link
                        to="/dashboard"
                        className="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 md:py-4 md:text-lg md:px-10 transition-all transform hover:scale-105"
                      >
                        Start Detecting
                      </Link>
                    </div>
                    <div className="mt-3 sm:mt-0 sm:ml-3">
                      <Link
                        to="/dashboard"
                        className="w-full flex items-center justify-center px-8 py-3 border border-gray-700 text-base font-medium rounded-md text-gray-300 bg-gray-800 hover:bg-gray-700 md:py-4 md:text-lg md:px-10 transition-all"
                      >
                        Try Demo
                      </Link>
                    </div>
                  </div>
                </motion.div>

                {/* Right Logo/Image */}
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                  className="mt-10 lg:mt-0 relative"
                >
                  <div className="relative mx-auto w-full rounded-lg shadow-lg ring-1 ring-black ring-opacity-5 lg:max-w-md">
                    <div className="relative w-full h-96 bg-gradient-to-br from-blue-500/10 to-purple-500/10 rounded-lg overflow-hidden">
                      <img
                        src="/image1.png"
                        alt="Deepfake Detection AI"
                        className="w-full h-full object-cover"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-gray-900/70 via-gray-900/50 to-transparent mix-blend-multiply"></div>
                      <div className="absolute inset-0 bg-[#00a2ff]/20 mix-blend-color"></div>
                      <div className="absolute inset-0">
                        <div className="h-full w-full flex items-center justify-center">
                          <div className="grid grid-cols-4 gap-2 opacity-60">
                            {[...Array(16)].map((_, i) => (
                              <div key={i} className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" style={{ animationDelay: `${i * 0.1}s` }}></div>
                            ))}
                          </div>
                        </div>
                      </div>
                      <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-gray-900 to-transparent">
                        <h3 className="text-xl font-semibold text-white">AI-Powered Face Detection</h3>
                        <p className="mt-2 text-sm text-gray-300">Advanced facial analysis and deepfake detection</p>
                      </div>
                    </div>
                  </div>
                </motion.div>
              </div>
            </main>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-12 bg-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:text-center">
            <h2 className="text-base text-blue-400 font-semibold tracking-wide uppercase">Features</h2>
            <p className="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-white sm:text-4xl">
              Why Choose Our Deepfake Detector?
            </p>
          </div>

          <div className="mt-10">
            <div className="space-y-10 md:space-y-0 md:grid md:grid-cols-2 md:gap-x-8 md:gap-y-10">
              {features.map((feature, index) => (
                <motion.div
                  key={feature.name}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="relative bg-gray-800 p-6 rounded-lg border border-gray-700 hover:border-blue-500 transition-colors"
                >
                  <div className="absolute flex items-center justify-center h-12 w-12 rounded-md bg-gradient-to-r from-blue-500 to-purple-500 text-white">
                    {feature.icon}
                  </div>
                  <p className="ml-16 text-lg leading-6 font-medium text-white">{feature.name}</p>
                  <p className="mt-2 ml-16 text-base text-gray-400">{feature.description}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="bg-gray-900 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="text-center bg-gray-800 p-6 rounded-lg border border-gray-700"
              >
                <div className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500">
                  {stat.value}
                </div>
                <div className="mt-2 text-sm text-gray-400">{stat.name}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

const features = [
  {
    name: 'Advanced AI Detection',
    description: 'Our state-of-the-art AI model can accurately detect deepfakes with high precision.',
    icon: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
  },
  {
    name: 'Real-time Analysis',
    description: 'Get instant results with our fast and efficient deepfake detection system.',
    icon: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
  },
  {
    name: 'Secure Platform',
    description: 'Your data is protected with enterprise-grade security measures.',
    icon: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
      </svg>
    ),
  },
  {
    name: 'User Dashboard',
    description: 'Track and manage your deepfake detection history with our intuitive dashboard.',
    icon: (
      <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
  },
];

const stats = [
  { name: 'Happy Clients', value: '1000+' },
  { name: 'Deepfakes Detected', value: '15000+' },
  { name: 'Dedicated Staff', value: '10+' },
  { name: 'Businesses', value: '8+' },
];

export default Home; 