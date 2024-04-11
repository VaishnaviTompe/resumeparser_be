
// const mongoose = require('mongoose');
// const PDFParser = require('pdf-parse');

// // Define the schema for the Resume
// const resumeSchema = new mongoose.Schema({
//     userId: String,
//     firstName: String,
//     lastName: String,
//     address: String,
//     email: String,
//     contact: String,
//     resume: {
//         data: Buffer, // Assuming resume data is stored as a Buffer
//         contentType: String // Mime type of the resume data
//     },
//     resumeText: String 
// });

// // Middleware to convert PDF data to plain text before saving
// resumeSchema.pre('save', async function(next) {
//     console.log('Resume Data:', this.resume);
//     try {
//         if (!this.resumeText && this.resume && this.resume.data) {
//             const pdfBuffer = this.resume.data;
//             const pdfText = await PDFParser(pdfBuffer);
//             this.resumeText = pdfText.text;
//             console.log('Resume Text:', this.resumeText);
//         }
//     } catch (error) {
//         console.error('Error parsing PDF:', error);
//         throw new Error('Could not parse PDF');
//     }
//     next();
// });


//   const Resume = mongoose.model('Resume', resumeSchema);

// module.exports = mongoose.model('Resume', resumeSchema);



const mongoose = require('mongoose');
const PDFParser = require('pdf-parse');

// Import the UserModel schema
const UserModel = require('./user_model');

// Define the schema for the Resume
const resumeSchema = new mongoose.Schema({
    // Reference to the UserModel schema
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'UserModel',
        required: true
    },
    fullName: {
        type: String,
        required: true
    },
    email: {
        type: String,
        required: true
    },
    resume: {
        data: Buffer, // Assuming resume data is stored as a Buffer
        contentType: String // Mime type of the resume data
    },
    resumeText: String 
});

// Middleware to convert PDF data to plain text before saving
resumeSchema.pre('save', async function(next) {
    console.log('Resume Data:', this.resume);
    try {
        if (!this.resumeText && this.resume && this.resume.data) {
            const pdfBuffer = this.resume.data;
            const pdfText = await PDFParser(pdfBuffer);
            this.resumeText = pdfText.text;
            console.log('Resume Text:', this.resumeText);
        }
    } catch (error) {
        console.error('Error parsing PDF:', error);
        throw new Error('Could not parse PDF');
    }
    next();
});


const Resume = mongoose.model('Resume', resumeSchema);

module.exports = Resume;
