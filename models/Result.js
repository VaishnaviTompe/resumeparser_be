
// const mongoose = require('mongoose');

// // Define the schema for the Resume
// const resultSchema = new mongoose.Schema({
//     userId: String,
//     firstName: String,
//     lastName: String,
//     email: String,
//     Accuracy: String,
//     totalQuestionsAsked: Number
// });


// module.exports = mongoose.model('Result', resultSchema);



const mongoose = require('mongoose');
const UserModel = require('./user_model'); // Import the UserModel

// Define the schema for the Result
const resultSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'UserModel', // Reference to the UserModel
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
    Accuracy: String,
    totalQuestionsAsked: Number
});

module.exports = mongoose.model('Result', resultSchema);
