
import { onAuthStateChanged } from 'firebase/auth';
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { auth } from './firebase';

const PrivateRoute = ({ children, ...rest }) => {
  const [loading, setLoading] = useState(true);
  const [signedIn, setSignedIn] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (user) => {
      if (user) {
        // User is signed in
        const uid = user.uid;
        console.log("User #" + uid + " is signed in");
        setSignedIn(true);
        setLoading(false);
      } else {
        // User is signed out
        console.log("User is signed out");
        setSignedIn(false);
        setLoading(false);
      }
    });

    // Cleanup subscription on unmount
    return () => unsubscribe();
  }, []);

  useEffect(() => {
    console.log("signedInState: " + signedIn);
    if (!signedIn && !loading) {
      navigate('/signin');
    }
  }, [signedIn, navigate, loading]);

  if (loading) {
    return (<p>Loading...</p>);  // Or return a loading spinner
  }

  return signedIn ? children : null;
};

export default PrivateRoute;
