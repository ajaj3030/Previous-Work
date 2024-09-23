// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getFunctions, httpsCallable } from "firebase/functions";

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
	apiKey: "AIzaSyD6_OtVUxFkY-XqEjRz-qBMB6VgVEXn1Mo",
	authDomain: "thepinfactory-42d2a.firebaseapp.com",
	projectId: "thepinfactory-42d2a",
	storageBucket: "thepinfactory-42d2a.appspot.com",
	messagingSenderId: "298424107673",
	appId: "1:298424107673:web:9d3060dcb2b234a150ee08",
	measurementId: "G-8SDMJF17BF",
};

// Initialize Firebase
const firebase = initializeApp(firebaseConfig);

// Export the auth and analytics services for easy access
const analytics = getAnalytics(firebase);
const auth = getAuth(firebase);
const db = getFirestore(firebase);
const functions = getFunctions(firebase);

export { auth, analytics, db, functions, httpsCallable };
