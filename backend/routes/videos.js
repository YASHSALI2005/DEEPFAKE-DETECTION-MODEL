const express = require('express');
const multer = require('multer');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const router = express.Router();

// Configure multer for video upload
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = path.join(__dirname, '../ml/test_videos');
    // Create directory if it doesn't exist
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    cb(null, file.originalname);
  }
});

const upload = multer({ 
  storage: storage,
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo'];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Only MP4, MOV, and AVI files are allowed.'));
    }
  }
});

// Direct video test endpoint
router.post('/test', upload.single('video'), async (req, res) => {
  try {
    // Validate video file
    if (!req.file) {
      return res.status(400).json({ error: 'No video file uploaded' });
    }

    // Check if file is a valid video
    const videoPath = req.file.path;
    const videoStats = fs.statSync(videoPath);
    if (videoStats.size === 0) {
      fs.unlinkSync(videoPath);
      return res.status(400).json({ error: 'Uploaded file is empty' });
    }

    // Run Python script
    const pythonProcess = spawn('python', [
      path.join(__dirname, '..', 'ml', 'test_model.py'),
      videoPath
    ]);

    let pythonData = '';
    let pythonError = '';

    pythonProcess.stdout.on('data', (data) => {
      pythonData += data.toString();
      // Log performance metrics to console
      const output = data.toString();
      if (output.includes('Performance Metrics:')) {
        console.log('\n=== Performance Metrics ===');
        console.log(output);
        console.log('========================\n');
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      pythonError += data.toString();
      console.error('Python error:', data.toString());
    });

    pythonProcess.on('close', async (code) => {
      console.log('Python process exited with code:', code);

      if (code !== 0) {
        // Clean up on error
        try {
          if (fs.existsSync(videoPath)) {
            fs.unlinkSync(videoPath);
          }
        } catch (err) {
          console.error('Error deleting uploaded file:', err);
        }
        return res.status(500).json({ 
          error: 'Error processing video',
          details: pythonError
        });
      }

      // Parse the Python script output
      const output = pythonData + pythonError;
      const verdictMatch = output.match(/VERDICT:(FAKE|REAL)/);
      const scoreMatch = output.match(/SCORE:([\d.]+)/);
      const sampleFacesDirMatch = output.match(/SAMPLE_FACES_DIR:(.+)/);
      const sampleFacesMatch = output.match(/SAMPLE_FACES:(.+)/);
      const metricsMatch = output.match(/Performance Metrics:\s+([\s\S]+?)(?=\n\n|\n$)/);
      
      if (!verdictMatch || !scoreMatch) {
        // Clean up on parsing error
        try {
          if (fs.existsSync(videoPath)) {
            fs.unlinkSync(videoPath);
          }
        } catch (err) {
          console.error('Error deleting uploaded file:', err);
        }
        return res.status(500).json({
          error: 'Could not parse model output',
          details: output
        });
      }

      const verdict = verdictMatch[1];
      const score = parseFloat(scoreMatch[1]);
      
      // Get sample faces if available
      let sampleFaces = [];
      if (sampleFacesDirMatch && sampleFacesMatch) {
        const sampleFacesDir = sampleFacesDirMatch[1];
        const sampleFacesList = sampleFacesMatch[1].split(',');
        
        try {
          sampleFaces = sampleFacesList.map(faceFile => {
            const facePath = path.join(sampleFacesDir, faceFile);
            if (!fs.existsSync(facePath)) {
              console.error('Face file not found:', facePath);
              return null;
            }
            const faceData = fs.readFileSync(facePath);
            return faceData.toString('base64');
          }).filter(face => face !== null);
        } catch (error) {
          console.error('Error processing sample faces:', error);
        }
      }

      // Parse metrics if available
      let metrics = null;
      if (metricsMatch) {
        const metricsText = metricsMatch[1];
        metrics = {
          precision: parseFloat(metricsText.match(/Precision: ([\d.]+)/)[1]),
          recall: parseFloat(metricsText.match(/Recall: ([\d.]+)/)[1]),
          f1_score: parseFloat(metricsText.match(/F1 Score: ([\d.]+)/)[1]),
          accuracy: parseFloat(metricsText.match(/Accuracy: ([\d.]+)/)[1])
        };
      }

      const response = {
        results: {
          verdict: verdict,
          score: score,
          metrics: metrics,
          sampleFaces: sampleFaces,
          details: `Video analyzed with ${verdict.toLowerCase()} confidence of ${(score * 100).toFixed(1)}%`
        }
      };
      
      // Send response first
      res.json(response);

      // Clean up files after sending response
      setTimeout(() => {
        try {
          // Delete the uploaded video
          if (fs.existsSync(videoPath)) {
            fs.unlinkSync(videoPath);
            console.log('Successfully deleted uploaded video');
          }

          // Delete the sample faces directory if it exists
          if (sampleFacesDirMatch) {
            const sampleFacesDir = sampleFacesDirMatch[1];
            if (fs.existsSync(sampleFacesDir)) {
              fs.rmSync(sampleFacesDir, { recursive: true, force: true });
              console.log('Successfully deleted sample faces directory');
            }
          }
        } catch (err) {
          console.error('Error during cleanup:', err);
        }
      }, 2000); // Wait 2 seconds before cleanup
    });
  } catch (err) {
    console.error('Error processing video:', err);
    res.status(500).json({ error: 'Error processing video' });
  }
});

module.exports = router; 