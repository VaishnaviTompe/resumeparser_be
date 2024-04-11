// const express = require('express');
// const router = express.Router();
// const User = require('../models/user_model');
// const Resume = require('../models/resume');
// const UserQuestions = require('../models/UserQuestions');
// const authMiddleware = require('../middleware/authmiddleware');
// const multer = require('multer');

// const { PDFLoader } = require("langchain/document_loaders/fs/pdf");
// const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
// const { CohereEmbeddings } = require("@langchain/cohere");
// const { HNSWLib } = require("@langchain/community/vectorstores/hnswlib");
// const { PromptTemplate } = require("@langchain/core/prompts");
// const { StringOutputParser } = require("@langchain/core/output_parsers");
// const {
//   RunnablePassthrough,
//   RunnableSequence,
// } = require("@langchain/core/runnables");
// const { Document } = require("@langchain/core/documents");
// const { ChatAnthropic } = require("@langchain/anthropic");

// const COHERE_API_KEY = process.env.COHERE_API_KEY;
// const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;

// const upload = multer({ dest: 'uploads/' });

// router.post('/submit-resume', authMiddleware, upload.single('resume'), async (req, res) => {
//     let filePath;
//     try {
//       // Handling file upload
//       if (!req.file) {
//         return res.status(400).json({ message: 'No resume file uploaded' });
//       }
//       filePath = req.file.path;
  
//       // Load and process the resume file
//       const loader = new PDFLoader(filePath);
//       const docs = await loader.load();
  
//       const splitter = new RecursiveCharacterTextSplitter({
//         chunkSize: 1000,
//         chunkOverlap: 20,
//       });
  
//       const splittedDocs = await splitter.splitDocuments(docs);
  
//       // Get the userId from the request body
//       const userId = req.body.userId;
  
//       if (!userId) {
//         return res.status(400).json({ message: 'User ID is required' });
//       }
  
//       // Store the processed resume in the database
//       const fileContent = fs.readFileSync(filePath);
//       const newResume = new Resume({
//         userId: userId,
//         firstName: req.body.firstName,
//         lastName: req.body.lastName,
//         address: req.body.address,
//         email: req.body.email,
//         contact: req.body.contact,
//         resume: {
//           data: fileContent,
//           contentType: req.file.mimetype
//         },
//         // You may want to store additional information from the resume as needed
//       });
  
//       await newResume.save();
  
//       res.status(201).json({ message: 'Resume submitted successfully' });
//     } catch (error) {
//       console.error(error);
//       res.status(500).json({ error: 'Internal Server Error' });
//     } finally {
//       // Cleanup: delete the uploaded file
//       if (filePath) {
//         fs.unlinkSync(filePath);
//       }
//     }
//   });

//   router.post('/ask-question', authMiddleware, async (req, res) => {
//     try {
//       const { userId, question } = req.body;
  
//       console.log('Request Body:', req.body);
  
//       if (!userId || !question) {
//         return res.status(400).json({ message: 'User ID and Question is required' });
//       }
  
//       const resume = await Resume.findOne({ userId: userId }); 
//       if (!resume) {
//         return res.status(404).json({ message: 'No resume found' });
//       }
  
//       let answer;
  
//       // Assuming `resumeText` is the field where the text from the resume is stored
//       const resumeText = resume.resumeText;
  
//       const vectorstore = await HNSWLib.fromDocuments(
//         [{ pageContent: resumeText, metadata: {} }],
//         new CohereEmbeddings({ apiKey: process.env.COHERE_API_KEY })
//       );
//       const retriever = vectorstore.asRetriever();
  
//       const model = new ChatAnthropic(process.env.ANTHROPIC_API_KEY);
//       const template = `Answer the question based only on the following context:
//       {context}
  
//       Question: {question}`;
//       const prompt = PromptTemplate.fromTemplate(template);
//       const formatDocs = (docs) => docs.map((doc) => doc.pageContent);
//       const retrievalChain = RunnableSequence.from([
//         { context: retriever.pipe(formatDocs), question: new RunnablePassthrough() },
//         prompt,
//         model,
//         new StringOutputParser(),
//       ]);
//       answer = await retrievalChain.invoke(question);
  
//       const newUserQuestions = new UserQuestions({
//         userId: userId,
//         question: question,
//         answer: answer
//       });
  
//       await newUserQuestions.save(); 
  
//       console.log('Answer:', answer);
//       res.status(200).json({ answer });
//     } catch (error) {
//       console.error(error);
//       res.status(500).json({ error: 'Internal Server Error' });
//     }
//   });

// // Define the route with auth middleware
// router.get('/shortlist-candidates', authMiddleware, async (req, res) => {
//     try {
//       // Placeholder logic for fetching resumes from the database (Replace with actual logic)
//       const resumes = await Resume.find();
  
//       // Define the minimum accuracy threshold required for shortlisting (e.g., 60%)
//       const MINIMUM_ACCURACY_THRESHOLD = 0.6;
  
//       // Array to store shortlisted candidates
//       const shortlistedCandidates = [];
  
//       // Iterate through each resume
//       for (const resume of resumes) {
//         // Fetch all user questions for the current resume
//         const userQuestions = await UserQuestions.find({ userId: resume.userId });
  
//         // Calculate the number of correct answers received
//         const correctAnswersCount = userQuestions.reduce((acc, curr) => {
//           // Assuming answers are considered correct if they are not empty strings
//           return acc + (curr.answer.trim() !== '' ? 1 : 0);
//         }, 0);
  
//         // Calculate the accuracy
//         const accuracy = correctAnswersCount / userQuestions.length;
  
//         // Check if the accuracy meets or exceeds the minimum threshold
//         if (accuracy >= MINIMUM_ACCURACY_THRESHOLD) {
//           // If yes, add the candidate to the shortlisted array
//           shortlistedCandidates.push({
//             userId: resume.userId,
//             firstName: resume.firstName,
//             lastName: resume.lastName,
//             email: resume.email,
//             accuracy: accuracy * 100,
//             totalQuestionsAsked: userQuestions.length
//           });
//         }
//       }
  
//       // Return the shortlisted candidates
//       res.status(200).json(shortlistedCandidates);
//     } catch (error) {
//       console.error(error);
//       res.status(500).json({ error: 'Internal Server Error' });
//     }
//   });
  
  

// module.exports = router;
