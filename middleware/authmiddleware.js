// authmiddleware.js 

const jwt = require("jsonwebtoken");
require("dotenv").config();
const JWT_SECRET = process.env.JWT_SECRET;

const mongoose = require("mongoose");
const UserModel = require("../models/user_model");

module.exports = (req, res, next) => {
  const { authorization } = req.headers;
   console.log("Authorization Header:", authorization);

  if (!authorization) {
    console.log("Authorization header not found");
    return res.status(401).json({ error: "User not logged in" });
  }
  const token = authorization.replace("Bearer ", "").trim();
  console.log("Token:", token);

  jwt.verify(token, JWT_SECRET, (error, payload) => {
    // console.log(token);
    if (error) {
      console.error("Token verification error:", error);
      if (error.name === "TokenExpiredError") {
        return res.status(401).json({ error: "Token expired" });
      } else {
        console.error(error);
        return res.status(401).json({ error: "User not logged in" });
      }
    }
    
    const { _id } = payload;

    UserModel.findById(_id)
      .then((dbUser) => {
        if (!dbUser) {
          console.log("User not found in database");
          return res.status(401).json({ error: "User not found" });
        }
        req.user = dbUser;
        console.log("User found:", dbUser);
        next();
      })
      .catch((err) => {
        console.error("Error finding user in database:", err);
        res.status(500).json({ error: "Internal Server Error" });
      });
  });
};
