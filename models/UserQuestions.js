// const mongoose = require('mongoose');
// const PDFParser = require('pdf-parse');

// const userQuestionsSchema = new mongoose.Schema({
//     userId: String,
//     question: String,
//     answer: String
//   });

// //   const UserQuestions = mongoose.model('UserQuestions', userQuestionsSchema);

//   module.exports = mongoose.model('UserQuestions', userQuestionsSchema);


const mongoose = require('mongoose');

const userQuestionsSchema = new mongoose.Schema({
    userId: String,
    questions: [{
        question: String,
        answer: String
    }]
});

module.exports = mongoose.model('UserQuestions', userQuestionsSchema);
