const express = require("express");
const router = express.Router();
const bcryptjs = require("bcryptjs");
const jwt = require("jsonwebtoken");
const mongoose = require("mongoose");
const UserModel = mongoose.model("UserModel");
require("dotenv").config();
// const { JWT_SECRET } = require("../config"); 
const JWT_SECRET = process.env.JWT_SECRET;

// Route to handle user signup
router.post("/signup", (req, res) => {
  const { fullName, email, password } = req.body;

  if (!fullName || !password || !email) {
    return res
      .status(400)
      .json({ error: "One or more mandatory fields are empty" });
  }

  UserModel.findOne({ email: email })
    .then((userInDB) => {
      if (userInDB) {
        return res
          .status(500)
          .json({ error: "User with this email already registered" });
      }

      bcryptjs
        .hash(password, 16)
        .then((hashedPassword) => {
          const user = new UserModel({
            fullName,
            email,
            password: hashedPassword,
          });
          user
            .save()
            .then((newUser) => {
              res.status(201).json({ result: "User signed up successfully" });
            })
            .catch((error) => {
              console.log(error);
            });
        })
        .catch((error) => {
          console.log(error);
        });
    })
    .catch((error) => {
      console.log(error);
    });
});

// Route to handle user login
router.post("/login", (req, res) => {
  const { email, password } = req.body;

  if (!password || !email) {
    return res
      .status(400)
      .json({ error: "One or more mandatory fields are empty" });
  }

  UserModel.findOne({ email: email })
    .then((userInDB) => {
      if (!userInDB) {
        return res.status(401).json({ error: "Invalid credientials" });
      }

      bcryptjs
        .compare(password, userInDB.password)
        .then((didMatch) => {
          if (didMatch) {
            const jwtToken = jwt.sign({ _id: userInDB._id }, JWT_SECRET);
            const userInfo = {
              email: userInDB.email,
              fullName: userInDB.fullName,
            };
            res
              .status(200)
              .json({ result: { token: jwtToken, user: userInfo } });
          } else {
            return res.status(401).json({ error: "Invalid credientials" });
          }
        })
        .catch((error) => {
          console.log(error);
        });
    })
    .catch((error) => {
      console.log(error);
    });
});

module.exports = router;
