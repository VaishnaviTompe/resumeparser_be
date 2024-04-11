// Import the mongoose library
const mongoose = require("mongoose");

// Define a schema for the 'UserModel' model
const userSchema = new mongoose.Schema({
  fullName: {
    type: String,
    required: true,
  },

  email: {
    type: String,
    required: true,
  },

  password: {
    type: String,
    required: true,
  },
});

// Create a model 'UserModel' using the defined schema
const UserModel = mongoose.model("UserModel", userSchema);

// Export the 'UserModel' model
module.exports = UserModel;
