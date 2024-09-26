import React, { useState, Fragment } from 'react';

import { Listbox, Transition } from '@headlessui/react';
import { CloudArrowUpIcon, LockClosedIcon, ServerIcon, ChevronUpDownIcon } from '@heroicons/react/20/solid'
import { Switch } from '@headlessui/react'

import AvatarController from './AvatarAnimation/AvatarController';
import PhoneGeographies from '../../assets/geo/dic.js';

export default function PressInquiriesPage() {

    const [values, setValues] = useState({});
    const [phonePlaceholder, setPhonePlaceholder] = useState('Enter your mobile number');
    const [selectedPhoneCountry, setSelectedPhoneCountry] = useState(Object.values(PhoneGeographies)[0])
    const [isCompany, setIsCompany] = useState(false)

    const handleInputChange = (event) => {
        const { name, value } = event.target;
        setValues(prevValues => ({...prevValues, [name]: value }));
    };

  return (
    <div className="relative isolate bg-white">
      <div className="mx-auto grid max-w-7xl grid-cols-1 lg:grid-cols-2">
        
        {/* LEFT SIDE OF CONTENT */}
        <div className='h-screen'>

          {/* BACKGROUND */}
            <div className="fixed inset-y-0 left-0 -z-10 w-full h-screen overflow-hidden bg-gray-100 ring-1 ring-gray-900/10 lg:w-1/2">
                <svg
                    className="absolute inset-0 h-full w-full stroke-gray-200 [mask-image:radial-gradient(100%_100%_at_top_right,white,transparent)]"
                    aria-hidden="true"
                >
                <defs>
                  <pattern
                    id="83fd4e5a-9d52-42fc-97b6-718e5d7ee527"
                    width={200}
                    height={200}
                    x="100%"
                    y={-1}
                    patternUnits="userSpaceOnUse"
                  >
                    <path d="M130 200V.5M.5 .5H200" fill="none" />
                  </pattern>
                </defs>
                <rect width="100%" height="100%" strokeWidth={0} fill="white" />
                <svg x="100%" y={-1} className="overflow-visible fill-gray-50">
                  <path d="M-470.5 0h201v201h-201Z" strokeWidth={0} />
                </svg>
                <rect width="100%" height="100%" strokeWidth={0} fill="url(#83fd4e5a-9d52-42fc-97b6-718e5d7ee527)" />
              </svg>
            </div>

            {/* LEFT SIDE CONTENT */}
            <div className='fixed inset-y-0 left-0 w-1/2 h-screen overflow-hidden'>
                <div className="px-6 lg:px-8 mx-auto max-w-xl">
                    <AvatarController />
                </div>
            </div>
        </div>

        {/* RIGHT SIDE OF CONTENT */}
        <form action="#" method="POST" className="relative z-10 px-6 pb-24 sm:pb-32 lg:px-8 lg:py-16">
            <div className="mx-auto max-w-xl lg:mr-0 lg:max-w-lg">

                <div className='mb-14'>
                    <h2 className="text-3xl font-bold tracking-tight text-gray-900">Press Inquiries</h2>
                    <p className="mt-2 text-md leading-6 text-gray-600">
                        Send any press enquiries here, and our team will be in touch.
                    </p>
                </div>

                <div className="grid grid-cols-1 gap-x-8 gap-y-6 sm:grid-cols-2">

                    {/* START FIRST NAME */}
                    <div>
                        <label htmlFor="first-name" className="block text-sm font-semibold leading-6 text-gray-900">
                            First name
                        </label>
                        <div className="mt-1.5">
                            <input
                            type="text"
                            id="first_name"
                            placeholder='Alice'

                            name="first_name"
                            autoComplete="given-name"
                            value={values.firstName}

                            onChange={handleInputChange}
                            className={`block w-full ${values.first_name ? "bg-blue-100/70 focus:bg-gray-100" : "bg-gray-100"} rounded-md px-3.5 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-800/60 sm:text-sm sm:leading-6`}
                            />
                        </div>
                    </div>
                    {/* END FIRST NAME */}

                    {/* START LAST NAME */}
                    <div>
                        <label htmlFor="last-name" className="block text-sm font-semibold leading-6 text-gray-900">
                            Last name
                        </label>
                        <div className="mt-1.5">
                            <input
                            type="text"
                            id="last_name"
                            placeholder='Allison'

                            name="last_name"
                            autoComplete="given-name"
                            value={values.last_name}

                            onChange={handleInputChange}
                            className={`block w-full ${values.last_name ? "bg-blue-100/70 focus:bg-gray-100" : "bg-gray-100"} rounded-md px-3.5 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-800/60 sm:text-sm sm:leading-6`}
                            />
                        </div>
                    </div>
                    {/* END LAST NAME */}

                    {/* START EMAIL */}
                    <div className="sm:col-span-2">
                        <label htmlFor="email" className="block text-sm font-semibold leading-6 text-gray-900">
                            Email
                        </label>
                        <div className="mt-1.5">
                            <input
                            type="text"
                            id="email"
                            placeholder='alice@example.com'

                            name="email"
                            autoComplete="given-name"
                            value={values.email}

                            onChange={handleInputChange}
                            className={`block w-full ${values.email ? "bg-blue-100/70 focus:bg-gray-100" : "bg-gray-100"} rounded-md px-3.5 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-800/60 sm:text-sm sm:leading-6`}
                            />
                        </div>
                    </div>
                    {/* END EMAIL */}

                    {/* START PHONE */}
                    <div className="sm:col-span-2">
                        
                        <label htmlFor="phone" className="block text-sm font-semibold leading-6 text-gray-900">
                            Phone
                        </label>

                        <div className="relative mt-1.5">
                            {/* PHONE COUNTRY CODE AND DROP DOWN */}
                            <Listbox value={selectedPhoneCountry} onChange={setSelectedPhoneCountry}>
                                {({ open }) => (
                                    <>
                                <div className={`absolute inset-y-0 left-0 flex items-center rounded-md ${open ? "ring-2 ring-blue-800/60" : ""}`}>
                                    <div className='relative'>

                                        <Listbox.Button className={`relative w-24 mt-1 pl-3.5 cursor-default text-left text-gray-900 sm:text-sm sm:leading-6`}>
                                            <div className="flex items-center justify-between">
                                                <div className="flex items-center">
                                                    <img src={'/flags/' + selectedPhoneCountry.code.toLocaleUpperCase() + '.svg'} alt="Flag Image" className="h-5 w-5" />
                                                    <span className="ml-3 mr-2">{"+" + selectedPhoneCountry.phoneNumberPrefix}</span>
                                                </div>
                                                <ChevronUpDownIcon className="h-4 w-4 text-gray-900" aria-hidden="true" />
                                            </div>
                                        </Listbox.Button>

                                        <Transition
                                        show={open}
                                        as={Fragment}
                                        leave="transition ease-in duration-100"
                                        leaveFrom="opacity-100"
                                        leaveTo="opacity-0"
                                        >
                                        <Listbox.Options className="absolute z-10 mt-4 max-h-64 w-72 overflow-auto rounded-md bg-white py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
                                            {Object.values(PhoneGeographies).map((geo) => (
                                            <Listbox.Option
                                                key={geo.code}
                                                className={({ active }) =>
                                                classNames(
                                                    active ? 'rounded-md bg-gray-100' : 'text-gray-900',
                                                    'relative cursor-default select-none mx-1 py-2'
                                                )
                                                }
                                                value={geo}
                                            >
                                                {({ selected }) => (
                                                <>
                                                    <div className="flex justify-between mx-4">
                                                        <div className='flex items-center'>
                                                            {/* ADD FLAG HERE */}
                                                            <img src={'/flags/' + geo.code.toLocaleUpperCase() + '.svg'} alt="Flag Image" className="mr-4 h-5 w-5 rounded-full" />
                                                            <span className={classNames(selected ? 'font-semibold' : 'font-normal', 'block truncate')}>
                                                                {geo.name}
                                                            </span>
                                                        </div>
                                                        <span className={'font-normal block truncate text-gray-500'}>
                                                            +{geo.phoneNumberPrefix}
                                                        </span>
                                                    </div>

                                                </>
                                                )}
                                            </Listbox.Option>
                                            ))}
                                        </Listbox.Options>
                                        </Transition>
                                    </div>
                                </div>
                                </>
                            )}
                            </Listbox>
                                
                            {/* ACTUAL PHONE ENTRY */}
                            <input
                            type="tel"
                            id="phone"
                            placeholder={selectedPhoneCountry.examplePhoneNumber}

                            name="phone"
                            autoComplete="given-name"
                            value={values.phone}

                            onChange={handleInputChange}
                            className={`block w-full pl-28 ${values.phone ? "bg-blue-100/70 focus:bg-gray-100" : "bg-gray-100"} rounded-md px-3.5 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-800/60 sm:text-sm sm:leading-6`}
                            />
                        </div>
                    </div>
                    {/* END PHONE */}

                    {/* START COMPANY */}
                    <div className="sm:col-span-2">
                        <label htmlFor="company" className="block text-sm font-semibold leading-6 text-gray-900">
                            Affiliation
                        </label>
                        <div className="mt-1.5">
                            <input
                            type="text"
                            id="affiliation"
                            placeholder="The Sunday Times"

                            name="affiliation"
                            autoComplete="given-name"
                            value={values.affiliation}

                            onChange={handleInputChange}
                            className={`block w-full ${values.affiliation ? "bg-blue-100/70 focus:bg-gray-100" : "bg-gray-100"} rounded-md px-3.5 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-800/60 sm:text-sm sm:leading-6`}
                            />
                        </div>
                    </div>
                    
                    <div className="sm:col-span-2">
                        <div className="flex justify-between text-sm leading-6">
                            <label htmlFor="message" className="block text-sm font-semibold leading-6 text-gray-900">
                                Message
                            </label>
                            <p id="message-description" className="text-gray-400">
                            Max 500 characters
                            </p>
                        </div>
                        <div className="mt-1.5">
                            <textarea
                            type="message"
                            id="message"
                            placeholder='Add message here...'

                            name="message"
                            autoComplete="given-name"
                            rows={4}
                            value={values.message}
                            defaultValue={''}

                            onChange={handleInputChange}
                            className={`block w-full ${values.message ? "bg-blue-100/70 focus:bg-gray-100" : "bg-gray-100"} rounded-md px-3.5 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-800/60 sm:text-sm leading-6`}
                            />
                        </div>
                    </div>
                </div>
                <div className="mt-10">
                    <button
                    type="submit"
                    className="block w-full rounded-md bg-blue-950 px-3.5 py-2.5 text-center text-sm font-semibold text-white shadow-sm hover:bg-indigo-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
                    >
                    Send message
                    </button>
                </div>
                <p className="mt-4 text-sm leading-6 text-gray-500">
                    By submitting this form, I agree to the{' '}
                    <a href="#" className="font-semibold text-blue-950 hover:text-indigo-600">
                    privacy&nbsp;policy
                    </a>
                    .
                </p>
            </div>
        </form>
      </div>
    </div>
  )
}

function classNames(...classes) {
  return classes.filter(Boolean).join(' ')
}
