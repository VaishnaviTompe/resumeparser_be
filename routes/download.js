// app.get('/download-resume/:id', async (req, res) => {
//     try {
//       const resume = await Resume.findById(req.params.id);
//       if (!resume) {
//         return res.status(404).json({ message: 'Resume not found' });
//       }
  
//       // Convert resume data to base64
//       const resumeData = resume.data.buffer.toString('base64');
//       const resumeContentType = resume.contentType;
  
//       // Set content type and send resume data as response
//       res.set('Content-Type', resumeContentType);
//       res.send(Buffer.from(resumeData, 'base64'));
//     } catch (error) {
//       console.error('Error:', error);
//       res.status(500).json({ message: 'Internal server error' });
//     }
//   });
  