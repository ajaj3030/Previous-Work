import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import AgentRunAcrossController from "../Brochureware/AvatarAnimation/AgentRunAcrossController";
import { auth } from '../../firebase.js'
import { signInWithEmailAndPassword } from "firebase/auth";
import LoadingCircle from '../utils/LoadingCircle';
import { ExclamationCircleIcon, ArrowLeftIcon } from "@heroicons/react/20/solid";


export default function SignIn() {

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleInputChange = (event) => {
    const { name, value } = event.target;

    if (name === "email") {
      setEmail(value);
    } else if (name === "password") {
      setPassword(value);
    }
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLoading(true);
    try {
      const response = await signInWithEmailAndPassword(auth, email, password);
      // clear inputs and error on successful sign in
      console.log("Sign in successful");
      console.log(response.user.uid);
      setEmail('');
      setPassword('');
      setError(null);
      // Navigate to dashboard
      navigate('/threads');
    } catch (error) {
      console.log("Sign in failed");
      console.log(error);

      switch (error.code) {
        case 'auth/invalid-email':
          setError('Invalid email format.');
          break;
        case 'auth/user-disabled':
          setError('This user has been disabled.');
          break;
        case 'auth/user-not-found':
          setError('User not found.');
          break;
        case 'auth/wrong-password':
          setError('Incorrect password.');
          break;
        default:
          setError('Sign in failed.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    
    <div className="fixed top-0 left-0 right-0 bottom-0">
			{/* LEFT SIDE OF CONTENT */}
			<form
			onSubmit={handleSubmit}
			className="absolute bg-white z-10 top-0 left-0 w-1/2 h-screen overscroll-y-none overflow-y-scroll custom-scrollbar"
			>
				<div className="flex flex-col px-24 pt-12 space-y-8">

          <button onClick={() => navigate('/')}><ArrowLeftIcon className='w-6 text-gray-900 hover:opacity-70'/></button>

          <div>
						<h2 className="text-3xl font-bold tracking-tight text-gray-900">
							Sign in to your account
						</h2>
						<p className="mt-2 text-md leading-6 text-gray-600">
							Sign in to your account to continue.
						</p>
					</div>

          <div className="flex flex-col gap-y-4 my-4">
            {/* START EMAIL */}
            <div className="sm:col-span-2">
              <label
                htmlFor="email"
                className="block text-sm font-semibold leading-6 text-gray-900">
                Email
              </label>
              <div className="mt-1.5">
                <input
                  id="email"
                  name="email"
                  type="email"
                  value={email}
                  autoComplete="email"
                  placeholder='george@example.com'
                  onChange={handleInputChange}
                  className={`block w-full ${
                    email
                      ? "bg-blue-100/70 focus:bg-gray-100"
                      : "bg-gray-100"
                  } rounded-md px-3.5 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-800/60 sm:text-sm sm:leading-6`}
                />
              </div>
            </div>
            {/* END EMAIL */}

            {/* START PASSWORD */}
            <div className="sm:col-span-2">
              <label
                htmlFor="password"
                className="block text-sm font-semibold leading-6 text-gray-900">
                Password
              </label>
              <div className="mt-1.5">
                <input
                  id="password"
                  name="password"
                  type="password"
                  value={password}
                  onChange={handleInputChange}
                  autoComplete="current-password"
                  placeholder='Password'
                  className={`block w-full ${
                    password
                      ? "bg-blue-100/70 focus:bg-gray-100"
                      : "bg-gray-100"
                  } rounded-md px-3.5 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-800/60 sm:text-sm sm:leading-6`}
                />
              </div>
            </div>
            {/* END PASSWORD */}

            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <input
                  id="remember-me"
                  name="remember-me"
                  type="checkbox"
                  className="h-4 w-4 rounded border-gray-300 form-checkbox text-violet-800"
                />
                <label htmlFor="remember-me" className="ml-3 block text-sm leading-6 text-gray-900">
                  Remember me
                </label>
              </div>

              <div className="text-sm leading-6">
                <a href="#" className="font-semibold text-brand-sd hover:text-brand-hover">
                  Forgot password?
                </a>
              </div>
            </div>

            {/* ERROR */}
            <div className={`flex space-x-2 items-center justify-start mt-4 ${error ? '' : 'opacity-0'}`}>
              <ExclamationCircleIcon className="h-4 w-4 text-red-700" />
              <p className="text-start font-medium text-xs text-red-700">
                {error}
              </p>
            </div>
          </div>

          <div className="flex flex-col space-y-4 items-center">
            <button type="submit" disabled={loading} className="block w-full rounded-md bg-green-700 px-3.5 py-2.5 text-center text-sm font-semibold text-white shadow-sm hover:bg-green-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600">
              <div className="flex justify-between items-center">
                <div className="w-8" />
                  Sign in
                <div className="w-8">
                  {loading && ( <LoadingCircle size="5" color="white"/>)}
                </div>
              </div>
            </button>

            <p className="text-center text-sm leading-6 text-gray-500">
              Don't have an account?{' '}
              <Link to="/getstarted" className="font-semibold text-brand-sd hover:text-brand-hover">
                Sign up here
              </Link>
            </p>
          
          </div>
				</div>
			</form>

			{/* RIGHT SIDE */}
			<div className="absolute z-0 top-0 right-0 h-screen w-1/2 backdrop shadow-[inset_0_0px_50px_20px_#9ca3af]">
				{/* BACKGROUND */}
				<svg
				className="absolute inset-0 h-full w-full stroke-blue-900/30 [mask-image:radial-gradient(100%_100%_at_bottom_right,#FFFFFF80,#FFFFFF80)]"
				aria-hidden="true"
				>
					<defs>
						<pattern
							id="83fd4e5a-9d52-42fc-97b6-718e5d7ee527"
							width={200}
							height={200}
							x="100%"
							y={-1}
							patternUnits="userSpaceOnUse">
							<path
								d="M130 200V.5M.5 .5H200"
								strokeDasharray="6,6"
								strokeWidth="1.2"
								fill="none"
							/>
						</pattern>
					</defs>
					<rect
						width="100%"
						height="100%"
						strokeWidth={0}
						fill="#f9fafb"
					/>
					<svg
						x="100%"
						y={-1}
						className="overflow-visible fill-gray-100">
						<path
							d="M-470.5 0h201v201h-201Z"
							strokeWidth={0}
						/>
					</svg>
					<svg
						x="100%"
						y={-1}
						className="overflow-visible fill-gray-100">
						<path
							d="M-470.5 0h201v201h-201Z"
							strokeWidth={0}
						/>
					</svg>
					<rect
						width="100%"
						height="100%"
						strokeWidth={0}
						fill="url(#83fd4e5a-9d52-42fc-97b6-718e5d7ee527)"
					/>
				</svg>

				{/* Avatar Controller*/}
				<div className="absolute w-full h-full">
					<AgentRunAcrossController />
				</div>
			</div>
		</div>
  )
}
