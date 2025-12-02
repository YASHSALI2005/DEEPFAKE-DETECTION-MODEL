require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');

// Import routes
const videoRoutes = require('./routes/videos');

const app = express();

// Create necessary directories
const uploadsDir = path.join(__dirname, 'uploads');
const testVideosDir = path.join(__dirname, '../test_videos');
const testResultsDir = path.join(__dirname, '../test_results');

[uploadsDir, testVideosDir, testResultsDir].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// Middleware
const allowedOrigins = [
  'http://localhost:3000',
  'http://localhost:3001',
  'http://localhost:3003',
  'https://deepfake-detection-model.vercel.app',
  'https://deepfake-detection-model-jec28f3aq-yashs-projects-f0014021.vercel.app'
];

app.use(express.json());
app.use(cors({
  origin: allowedOrigins,
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));
app.use(express.urlencoded({ extended: true }));

// Serve uploaded files
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Routes
app.use('/api/videos', videoRoutes);

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ 
    message: 'Something went wrong!',
    error: process.env.NODE_ENV === 'development' ? err.message : undefined
  });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log('CORS enabled for:', allowedOrigins);
  console.log('Uploads directory:', uploadsDir);
  console.log('Test videos directory:', testVideosDir);
  console.log('Test results directory:', testResultsDir);
}); 