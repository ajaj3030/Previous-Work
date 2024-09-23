import React from 'react';

const ArrowHead = ({ className }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 12 24"
    className={`group-hover:translate-x-1.5 ${className} transition-all duration-200 ease-in-out transform`}
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M1.5 4.5L9 12l-7.5 7.5"
      strokeWidth={2.5}
      stroke="currentColor"
    />
  </svg>
);

const ArrowStalk = ({ className }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    className={`group-hover:scale-x-125 ${className} transition-all duration-200 ease-in-out transform origin-left`}
    
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M21 12H3"
      strokeWidth={2.5}
      stroke="currentColor"
    />
  </svg>
);

const LoginButton = () => (
  <button className="flex items-center hover:opacity-50 focus:outline-none group">
    <span className="px-1 text-m font-semibold">Log in</span>
    <div className="flex items-center">
      <ArrowStalk className="w-4 h-6" />
      <ArrowHead className="w-3 h-4 -ml-2.5" />
    </div>
  </button>
);

export { ArrowHead, ArrowStalk }
export default LoginButton;
