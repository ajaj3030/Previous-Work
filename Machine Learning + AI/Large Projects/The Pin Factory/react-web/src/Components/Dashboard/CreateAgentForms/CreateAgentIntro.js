import React, { useState, Fragment } from 'react';
import { CheckCircleIcon } from '@heroicons/react/24/solid'
import {
  FaceFrownIcon,
  FaceSmileIcon,
  FireIcon,
  HandThumbUpIcon,
  HeartIcon,
  PaperClipIcon,
  XMarkIcon,
} from '@heroicons/react/20/solid'
import { Listbox, Transition } from '@headlessui/react'

export default function CreateAgentIntro({ onNext, onPrev }) {

    return (
        <div className="relative z-10 my-12 w-full mx-24">
            <div className='mb-14'>
                <div className='mb-14'>
                    <h2 className="text-3xl font-bold tracking-tight text-gray-900">Create an agent</h2>
                    <div>
                        <p className="mt-2 text-md leading-6 text-gray-600">
                            What's involved in creating agents? Here's what you can do.
                        </p>
                    </div>
                </div>
                <div className='mb-14'>
                    <Graphic />
                </div>
                <p className="text-md leading-6 text-gray-600">
                    We'll walk you through each step as you go. Don't worry about not getting things right the first time around. You can always edit your agents later.
                </p>
            </div>
            <div>
                <button onClick={onNext} className={`mb-4 block w-full rounded-md bg-blue-950 px-3.5 py-2.5 text-center text-sm font-semibold text-white shadow-sm hover:bg-indigo-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600`}>
                    <div className="flex items-center justify-center">
                        Get started
                        {/* Loading SVG below can go here */}
                    </div>
                </button>
            </div>
        </div>
    );
};



// <button onClick={onPrev} className={`block w-full text-center text-sm text-black`}>
// <div className="flex items-center justify-center hover:underline hover:decoration-1 hover:underline-offset-2 hover:opacity-50">
//     Cancel
//     {/* Loading SVG below can go here */}
// </div>
// </button>

const steps = [
  { id: 0, title: 'Model.', text: 'Choose your root model from our selection.' },
  { id: 1, title: 'Avatar.', text: 'Give your agent some personality.' },
  { id: 2, title: 'Add memory.', text: "Upload what you'd like your agent to remember." },
  { id: 3, title: 'Train.', text: "Fine-tune your agent for specific tasks." },
  { id: 4, title: 'Scale.', text: "Add more agents for better execution of complex tasks." },
  { id: 5, title: 'Connect.', text: "Connect your agents to your existing accounts." },
//   { id: 7, title: 'Go.', text: "Start using your agents in your factory." },
]

function classNames(...classes) {
  return classes.filter(Boolean).join(' ')
}

function Graphic() {
  return (
    <div>
      <ul role="list" className="space-y-6">
        {steps.map((step, stepIdx) => (
          <li key={step.id} className="relative flex gap-x-4">
            <div
              className={classNames(
                stepIdx === steps.length - 1 ? 'h-6' : '-bottom-6',
                'absolute left-0 top-0 flex w-6 justify-center'
              )}
            >
              <div className="w-px bg-gray-200" />
            </div>
            <>
                <div className="relative flex h-6 w-6 flex-none items-center justify-center bg-white">
                    <div className="h-1.5 w-1.5 rounded-full bg-gray-100 ring-1 ring-gray-300" />
                </div>
                <p className="flex-auto py-0.5 text-sm leading-5 text-gray-500">
                  <span className="font-medium text-gray-900">{step.title}</span> {step.text}
                </p>
              </>
          </li>
        ))}
      </ul>
    </div>
  )
}






// Loading SVG
// {loading ? (
//     <div>
//         <svg
//             className="animate-spin h-4 w-4 text-white ml-3"
//             xmlns="http://www.w3.org/2000/svg"
//             fill="none"
//             viewBox="0 0 24 24"
//         >
//             <circle
//             className="opacity-25"
//             cx="12"
//             cy="12"
//             r="10"
//             stroke="currentColor"
//             strokeWidth="4"
//             ></circle>
//             <path
//             className="opacity-75"
//             fill="currentColor"
//             d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647zM20 12c0-3.042-1.135-5.824-3-7.938l-3 2.647A7.962 7.962 0 0120 12h4c0-6.627-5.373-12-12-12v4c3.042 0 5.824 1.135 7.938 3l-2.647 3z"
//             ></path>
//         </svg>
//     </div>
// ) : (<div/>)}