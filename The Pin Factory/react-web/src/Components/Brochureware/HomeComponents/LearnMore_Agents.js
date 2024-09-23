import React from 'react';

import { CloudArrowUpIcon, LockClosedIcon, ServerIcon } from '@heroicons/react/20/solid'

const features = [
  {
    name: 'Training.',
    description:
      "We do all the technical stuff for you. You don't even need any training data. Just tell us what you want your agent to do, and we'll do the rest.",
    icon: CloudArrowUpIcon,
  },
  {
    name: 'Memory.',
    description: 'Give your agents long-term memory. Academic papers? Legal documents? Financial reports? They can remember anything you want them to.',
    icon: LockClosedIcon,
  },
  {
    name: 'Plug-Ins.',
    description: "Connect your agents to your current infrastructure. We're always adding more, so if you don't see yours, let us know!",
    icon: ServerIcon,
  },
  {
    name: 'Communication.',
    description: "Create as many specialised agents as you want and have them collaborate to solve problems you didn't even know existed.",
    icon: ServerIcon,
  },
]

export default function LearnMore_Agents() {
  return (
    <div className="overflow-hidden py-24 sm:py-32 px-44">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto grid max-w-2xl grid-cols-1 gap-x-8 gap-y-16 sm:gap-y-20 lg:mx-0 lg:max-w-none lg:grid-cols-2">
          <div className="lg:pr-8 lg:pt-4">
            <div className="lg:max-w-lg">
              <h2 className="text-base font-semibold leading-7 text-indigo-600">An easier way to think of AI</h2>
              <p className="mt-2 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">Agents. What are they?</p>
              <p className="mt-6 text-lg leading-7 text-gray-600">
                Think of agents as your personal employees. The possibilities are endless.
              </p>
              <dl className="mt-10 max-w-xl space-y-8 text-base leading-7 text-gray-600 lg:max-w-none">
                {features.map((feature) => (
                  <div key={feature.name} className="relative pl-9">
                    <dt className="inline font-semibold text-gray-900">
                      <feature.icon className="absolute left-1 top-1 h-5 w-5 text-indigo-600" aria-hidden="true" />
                      {feature.name}
                    </dt>{' '}
                    <dd className="inline">{feature.description}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
          {/* <img
            src="https://tailwindui.com/img/component-images/dark-project-app-screenshot.png"
            alt="Product screenshot"
            className="w-[48rem] max-w-none rounded-xl shadow-xl ring-1 ring-gray-400/10 sm:w-[57rem] md:-ml-4 lg:-ml-0"
            width={2432}
            height={1442}
          /> */}
        </div>
      </div>
    </div>
  )
}

