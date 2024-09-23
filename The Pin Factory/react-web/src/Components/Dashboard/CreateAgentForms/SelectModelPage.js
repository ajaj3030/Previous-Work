import React, { useState, Fragment, useContext } from 'react';

import { AvatarWalkingForwards } from '../../Brochureware/AvatarAnimation/AvatarWalkForwards';
import { ModelContext } from './ModelContext';

import { Listbox, Transition } from '@headlessui/react'
import { CheckIcon, ChevronUpDownIcon, XMarkIcon, ArrowLeftIcon } from '@heroicons/react/20/solid'
import openai_logo from '../../../assets/openai-logos/PNGs/openai-logomark.png';
import google_logo from '../../../assets/google-logos/google-g-logo.png';

const { v4: uuidv4 } = require('uuid');

const models = [
    {
      id: 'gpt-3',
      name: 'GPT-3',
      avatar: openai_logo,
      status: 'Recommended',
      description: 'GPT-3 is a language model that uses deep learning to produce human-like text. It takes in a prompt, and attempts to complete it.',
      capabilities: {
          memory: true,
          training: true,
          communication: true,
          plugins: true,
      },
    },
    {
      id: 'gpt-4',
      name: 'GPT-4',
      avatar: openai_logo,
      status: '',
      description: 'GPT-4 is a language model that uses deep learning to produce human-like text. It takes in a prompt, and attempts to complete it.',
      capabilities: {
          memory: true,
          training: false,
          communication: true,
          plugins: true,
      },
    },
    {
      id: 'palm-2',
      name: 'PaLM 2',
      avatar: google_logo,
      status: 'Coming soon',
      description: 'PaLM 2 is a language model that uses deep learning to produce human-like text. It takes in a prompt, and attempts to complete it.',
      capabilities: {
          memory: true,
          training: true,
          communication: true,
          plugins: true,
      },
    },
]

const avatars = [
  {
    uid: uuidv4(),
    id: "BASIC",
    animated: <AvatarWalkingForwards status={"forwards"}/>,
    static: <AvatarWalkingForwards status={"still"}/>,
    sex: {
      id: "male",
      pronouns: {
        pronoun: "he",
        objective: "him",
        possessive: "his",
      }
    }
  },
  {
    uid: uuidv4(),
    id: "BASIC",
    animated: <AvatarWalkingForwards status={"forwards"}/>,
    static: <AvatarWalkingForwards status={"still"}/>,
    sex: {
      id: "male",
      pronouns: {
        pronoun: "he",
        objective: "him",
        possessive: "his",
      }
    }
  },
  {
    uid: uuidv4(),
    id: "BASIC",
    animated: <AvatarWalkingForwards status={"forwards"}/>,
    static: <AvatarWalkingForwards status={"still"}/>,
    sex: {
      id: "male",
      pronouns: {
        pronoun: "he",
        objective: "him",
        possessive: "his",
      }
    }
  },
  {
    uid: uuidv4(),
    id: "BASIC",
    animated: <AvatarWalkingForwards status={"forwards"}/>,
    static: <AvatarWalkingForwards status={"still"}/>,
    sex: {
      id: "male",
      pronouns: {
        pronoun: "he",
        objective: "him",
        possessive: "his",
      }
    }
  },
]


export default function SelectModelPage({ onNext, onPrev }) {

    const [selectedModel, setSelectedModel] = useState(models[0]);
    const [selectedAvatar, setSelectedAvatar] = useState(avatars[0]);
    const [name, setName] = useState(null);

    const [formSubmitted, setFormSubmitted] = useState(false);
    // const [formValid, setFormValid] = useState(false);

    const checkFormValidity = () => {
      let isValid = false;
      if ((name && name.trim().length > 0) && 
          selectedModel.id && selectedModel.id.trim().length > 0 && 
          selectedAvatar.id && selectedAvatar.id.trim().length > 0) {
        isValid = true;
      }
      return isValid;
    };

  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setName(value);
    checkFormValidity();
  };

  const { state, setState } = useContext(ModelContext);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setFormSubmitted(true);
    const formValid = checkFormValidity();
    if (formValid) {
      setState({ 
        agents: [...state.agents, 
          {
            id: uuidv4(),
            name: name,
            model: selectedModel,
            avatar: selectedAvatar,
            setup: {
              agent: "IN PROGRESS",
              memory: "NOT STARTED",
              training: "NOT STARTED",
              connect: "NOT STARTED",
            },
          }
        ],
        selectedAgent: null
      });
      onNext();
    } else {
      // Handle form isn't valid
      console.log("Form not valid");
    }
  };

  const handleBackClick = () => {
    onPrev();
  };
  
    return (
        <form action="#" method="POST" className="relative z-10 my-12 w-full mx-24" onSubmit={handleSubmit}>
          <div className='flex items-center justify-between mb-6'>
                <button onClick={handleBackClick}><ArrowLeftIcon className='w-6 text-gray-900 hover:opacity-70'/></button>
            </div>
            {/* Content container  */}
            <div className='mb-14'>
              {/* Header */}
              <div className='mb-8'>
                  <h2 className="text-3xl font-bold tracking-tight text-gray-900">Create your agent</h2>
                  <div>
                      <p className="mt-2 text-md leading-6 text-gray-600">
                          Let's build your agent base. Select your model. Choose an avatar. Give your agent a name.
                      </p>
                  </div>
              </div>

              {/* Model section */}
              <div className="sm:col-span-2 mb-12">
                  <label htmlFor="model" className="block text-sm font-semibold leading-6 text-gray-900">
                      Model
                  </label>
                  <div className="mb-4">
                    <ModelDropdown selectedModel={selectedModel} setSelectedModel={setSelectedModel} />
                  </div>
                  <ModelBreakdown selectedModel={selectedModel} />
              </div>

              {/* Avatar section */}
              <div className="sm:col-span-2 mb-12">
                  <label htmlFor="avatar" className="block text-sm font-semibold leading-6 text-gray-900">
                      Avatar
                  </label>
                  <p className="text-sm text-gray-500 mb-4">
                      Select your avatar for this agent.
                  </p>
                  <AvatarGrid selectedAvatar={selectedAvatar} setSelectedAvatar={setSelectedAvatar} />
              </div>

              {/* Name section */}
              <div className="sm:col-span-2 mb-12">
                  <label htmlFor="avatar" className="block text-sm font-semibold leading-6 text-gray-900">
                      Name
                  </label>
                  <p className="text-sm text-gray-500 mb-4">
                      What would you like your agent to be called?
                  </p>
                  <div className="mt-1.5">
                      <input
                      type="text"
                      id="name"
                      placeholder='Gregory Hirsch'
                      value={name}

                      onChange={handleInputChange}
                      className={`block w-full transition-all duration-100 ease-in-out ${
                          formSubmitted && !name ? "ring-1.5 ring-red-700" : ""
                      } ring-1 ring-gray-300 rounded-md px-3.5 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-800/60 sm:text-sm sm:leading-6`}
                      />
                      <p className={`mt-2 font-semibold text-xs text-red-700 transition-all duration-100 ease-in-out ${(formSubmitted && !name) ? '' : 'opacity-0'}`} id="email-error">
                        Please enter a name.
                      </p>
                      {/* {formSubmitted && !name && (
                        
                      )} */}
                  </div>
              </div>
              </div>

              {/* Button container */}
              <div>
                <button type="submit" onClick={handleSubmit} className={`mb-4 block w-full rounded-md bg-blue-950 px-3.5 py-2.5 text-center text-sm font-semibold text-white shadow-sm hover:bg-indigo-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600`}>
                  <div className="flex items-center justify-center">Create agent</div>
                  {/* Loading SVG below can go here */}
                </button>
                <button onClick={onPrev} className={`block w-full text-center text-sm text-black`}>
                    <div className="flex items-center justify-center hover:underline hover:decoration-1 hover:underline-offset-2 hover:opacity-50">
                        Cancel
                        {/* Loading SVG below can go here */}
                    </div>
                </button>
              </div>
        </form>
    );
};

function classNames(...classes) {
  return classes.filter(Boolean).join(' ')
}

function ModelDropdown({ selectedModel, setSelectedModel }) {
  return (
    <Listbox value={selectedModel} onChange={(model) => setSelectedModel(model)}>
      {({ open }) => (
        <>
          <div className="relative mt-2">
            <Listbox.Button className="relative w-full cursor-default rounded-md px-3.5 py-2 text-gray-900 ring-1 ring-inset ring-gray-300 outline-none focus:ring-2 focus:ring-blue-800/60 sm:text-sm sm:leading-6">
              <span className="flex items-center">
                <img src={selectedModel.avatar} alt="" className="h-5 w-5 flex-shrink-0 rounded-full" />
                <span className="ml-3 block truncate">{selectedModel.name}</span>
              </span>
              <span className="pointer-events-none absolute inset-y-0 right-0 ml-3 flex items-center pr-2">
                <ChevronUpDownIcon className="h-5 w-5 text-gray-400" aria-hidden="true" />
              </span>
            </Listbox.Button>

            <Transition
              show={open}
              as={Fragment}
              leave="transition ease-in duration-100"
              leaveFrom="opacity-100"
              leaveTo="opacity-0"
            >
              <Listbox.Options className="absolute z-10 mt-1 max-h-56 w-full overflow-auto rounded-md bg-white py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
                {models.map((model) => (
                  <Listbox.Option
                    key={model.id}
                    className={({ active }) =>
                      classNames(
                        active ? 'rounded-md bg-gray-100' : 'text-gray-900',
                        'relative cursor-default select-none mx-1 px-2 py-2', 
                        model.status == "Coming soon" ? "opacity-50" : ""
                      )
                    }
                    value={model}
                    disabled={model.status == "Coming soon"}
                    // onClick={}
                  >
                    {({ selectedModel, active }) => (
                      <>
                        <div className="flex items-center">
                          <img src={model.avatar} alt="" className="h-5 w-5 flex-shrink-0 rounded-full" />
                          <span
                            className={classNames(selectedModel ? 'font-semibold' : 'font-normal', 'ml-3 block truncate')}
                          >
                            {model.name}
                            {model.status && (
                                <span className={`ml-3 text-xs font-semibold ${model.status == "Recommended" ? "bg-purple-100 text-purple-500" : active ? "bg-gray-200/80 text-gray-500" : "bg-gray-100 text-gray-500"} rounded-md px-1.5 py-0.5`}>
                                    {model.status}
                                </span>
                            )}
                          </span>
                        </div>

                        {selectedModel ? (
                          <span
                            className={classNames(
                              'text-black absolute inset-y-0 right-0 flex items-center pr-4'
                            )}
                          >
                            <CheckIcon className="h-5 w-5" aria-hidden="true" />
                          </span>
                        ) : null}
                      </>
                    )}
                  </Listbox.Option>
                ))}
              </Listbox.Options>
            </Transition>
          </div>
        </>
      )}
    </Listbox>
  )
}


function ModelBreakdown({ selectedModel }) {
    return (
        <fieldset className="border-b border-gray-200">
            <div className="divide-y divide-gray-200">
                <div className="relative flex items-start pt-3 pb-3.5">
                    <div className="min-w-0 flex-1 text-sm leading-6">
                        <label htmlFor="memory" className="font-medium text-gray-800">
                            Memory
                        </label>
                        <p id="memory-description" className="text-gray-500">
                            Long-term memory allows to remember factual information.
                        </p>
                    </div>
                    <div className="ml-3 flex h-6 items-center">
                        {selectedModel.capabilities.memory ? (
                            <CheckIcon className="text-emerald-600 h-5 w-5" aria-hidden="true" />
                            ) : (
                            <XMarkIcon className="text-red-600 h-5 w-5" aria-hidden="true" />
                            )
                        }
                    </div>
                </div>
                <div className="relative flex items-start pb-4 pt-3.5">
                    <div className="min-w-0 flex-1 text-sm leading-6">
                        <label htmlFor="training" className="font-medium text-gray-800">
                            Training
                        </label>
                        <p id="training-description" className="text-gray-500">
                            Specialise model for specific tasks.
                        </p>
                    </div>
                    <div className="ml-3 flex h-6 items-center">
                        {selectedModel.capabilities.training ? (
                            <CheckIcon className="text-emerald-600 h-5 w-5" aria-hidden="true" />
                            ) : (
                            <XMarkIcon className="text-red-600 h-5 w-5" aria-hidden="true" />
                            )
                        }
                    </div>
                </div>
                <div className="relative flex items-start pb-4 pt-3.5">
                    <div className="min-w-0 flex-1 text-sm leading-6">
                        <label htmlFor="communication" className="font-medium text-gray-800">
                            Communication
                        </label>
                        <p id="communication-description" className="text-gray-500">
                            Model can communicate with other models to solve harder tasks.
                        </p>
                    </div>
                    <div className="ml-3 flex h-6 items-center">
                        {selectedModel.capabilities.communication ? (
                            <CheckIcon className="text-emerald-600 h-5 w-5" aria-hidden="true" />
                            ) : (
                            <XMarkIcon className="text-red-600 h-5 w-5" aria-hidden="true" />
                            )
                        }
                    </div>
                </div>
                <div className="relative flex items-start pb-4 pt-3.5">
                    <div className="min-w-0 flex-1 text-sm leading-6">
                        <label htmlFor="plugins" className="font-medium text-gray-800">
                            Plug-Ins
                        </label>
                        <p id="plugins-description" className="text-gray-500">
                            Plug-Ins let your agents execute tasks in the real world.
                        </p>
                    </div>
                    <div className="ml-3 flex h-6 items-center">
                        {selectedModel.capabilities.plugins ? (
                            <CheckIcon className="text-emerald-600 h-5 w-5" aria-hidden="true" />
                            ) : (
                            <XMarkIcon className="text-red-600 h-5 w-5" aria-hidden="true" />
                            )
                        }
                    </div>
                </div>
            </div>
        </fieldset>
    )
}

function AvatarGrid({ selectedAvatar, setSelectedAvatar }) {
  const handleChildButtonClick = (event, avatar) => {
    setSelectedAvatar(avatar);
  };
  const [hoveredIndex, setHoveredIndex] = useState(null);
  return (
    <ul role="list" className="grid grid-cols-2 gap-2">
      {avatars.map((avatar, index) => (
        <div
          key={index}
          onMouseEnter={() => setHoveredIndex(index)}
          onMouseLeave={() => setHoveredIndex(null)}
        >
          <button id="set_avatar" type="button" onClick={(e) => handleChildButtonClick(e, avatar)} className={`flex w-full py-2 items-center justify-center rounded-md ${selectedAvatar.uid==avatar.uid ? "ring-2 ring-blue-800/60" : "ring-1 hover:ring-2 ring-gray-300"}`}>
            {hoveredIndex === index ? (
              <div className='flex py-4 items-center justify-center rounded-md'>{avatar.animated}</div>
            ) : (
              <div className='flex py-4 items-center justify-center rounded-md'>{avatar.static}</div>
            )}
          </button>
        </div>
      ))}
    </ul>
  )
}
