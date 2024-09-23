import React, {useState, useEffect} from 'react';
import { ArrowStalk, ArrowHead } from '../Header/LogInButton.js';
import AvatarRunAcross from '../AvatarAnimation/AgentRunAcrossController.js';
import { Link } from 'react-router-dom';
import axios from 'axios';


const HomeLanding = () => {
  return (
    <div className="relative z-1">
      <div className="w-full h-full ring-1 ring-gray-900/10">
        <svg
        className="absolute inset-0 h-full w-full stroke-blue-900/30 [mask-image:radial-gradient(100%_100%_at_top_left,#FFFFFF80,#FFFFFF80)]"
        aria-hidden="true">
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
          className="overflow-visible fill-transparent">
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
      </div>
      <AvatarRunAcross yMarginOn={true} yMargin={76} className="z-0 absolute" />
      <div className="flex items-center justify-center h-screen bg-transparent relative z-1">
        <CentralHomeContent />
      </div>
    </div>

  );
};


const CentralHomeContent = () => {

    const homeStringFirstPart = "Intelligent Agents";
    const homeStringSecondPart = "";

    const handleClick = async () => {
    const promptData = { prompt: "how are you doing?" }; // replace with the prompt you want to send
    try {
      const res = await axios.post('http://18.170.45.114/api/hello/', promptData);
      console.log(res.data);
    } catch (error) {
      console.error(error);
    }
  };

    return (
        <div className="items-center bg-transparent">
        <h1 className="m-12 text-6xl text-center font-extrabold tracking-tight w-82">
            {homeStringFirstPart}
            <br/>
            {homeStringSecondPart}
        </h1>
        <div className="flex justify-center item-center gap-5">
            <GetStartedButton />
            {/* <button onClick={handleClick} className="py-2 px-5 flex items-center rounded-md bg-gray-100 hover:bg-gray-50 focus:outline-none group">
              <span className="px-1 text-base font-semibold">Learn more</span>
            </button> */}
        </div>
        </div>
    );
};

export const GetStartedButton = () => {

  const [message, setMessage] = useState('');

  const handleClick = () => {
    console.log('called');
    axios.get('http://127.0.0.1:5000/hello')
      .then(response => {
        setMessage(response.data.message);
        console.log("received");
      })
      .catch(error => {
        console.error('There was an error!', error);
      });
    console.log('finished');
  };

  return (

    <Link to="/agents">
      <div>
        <button onClick={handleClick} className="py-2 px-5 flex items-center justify-center rounded-md bg-green-700 text-white hover:bg-green-600 focus:outline-none group">
          <span className="px-1 text-base font-semibold">Get started</span>
          <div className="flex items-center">
              <ArrowStalk className="w-4 h-6" />
              <ArrowHead className="w-3 h-4 -ml-2.5" />
          </div>
        </button>
      </div>
    </Link>

  );
};

export default HomeLanding ;
