import React, { useState, Fragment } from 'react';

import { Listbox, Transition } from '@headlessui/react';
import { CloudArrowUpIcon, LockClosedIcon, ServerIcon, ChevronUpDownIcon } from '@heroicons/react/20/solid'
import { Switch } from '@headlessui/react'

import AvatarController from './AvatarAnimation/AvatarController';
import PhoneGeographies from '../../assets/geo/dic.js';

const ContactSalesBullets = [
    {
      name: 'Push to deploy.',
      description:
        'Lorem ipsum, dolor sit amet consectetur adipisicing elit. Maiores impedit perferendis suscipit eaque, iste dolor cupiditate blanditiis ratione.',
      icon: CloudArrowUpIcon,
    },
    {
      name: 'SSL certificates.',
      description: 'Anim aute id magna aliqua ad ad non deserunt sunt. Qui irure qui lorem cupidatat commodo.',
      icon: LockClosedIcon,
    },
    {
      name: 'Database backups.',
      description: 'Ac tincidunt sapien vehicula erat auctor pellentesque rhoncus. Et magna sit morbi lobortis.',
      icon: ServerIcon,
    },
  ]


export default function ContactSalesPage() {

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
                    <h2 className="text-3xl font-bold tracking-tight text-gray-900">Let's work together</h2>
                    <p className="mt-2 text-md leading-6 text-gray-600">
                        Contact our sales team here, and we'll get back to you right away.
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
                            placeholder='Freddie'

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
                            placeholder='Frank'

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
                            placeholder='freddie@example.com'

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
                    <div className="sm:col-span-2 my-4">
                        <label htmlFor="email" className="block text-sm font-semibold leading-6 text-gray-900">
                            Are you a company?
                        </label>
                        <Switch.Group as="div" className="flex items-center justify-between sm:col-span-2">
                            <span className="flex flex-grow flex-col">
                                <Switch.Label as="span" className={`text-sm leading-6 ${isCompany ? 'text-gray-900' : 'text-gray-400'}`} passive>
                                    Yes, I am contacting on behalf of a company.
                                </Switch.Label>
                                {/* <Switch.Description as="span" className="text-sm text-gray-500">
                                Nulla amet tempus sit accumsan. Aliquet turpis sed sit lacinia.
                                </Switch.Description> */}
                            </span>
                            <Switch
                                checked={isCompany}
                                onChange={setIsCompany}
                                className={classNames(
                                isCompany ? 'bg-blue-800' : 'bg-gray-200',
                                'relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none'
                                )}
                            >
                                <span
                                aria-hidden="true"
                                className={classNames(
                                    isCompany ? 'translate-x-5' : 'translate-x-0',
                                    'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out'
                                )}
                                />
                            </Switch>
                        </Switch.Group>

                        {isCompany && (
                            <div className='sm:col-span-2'>
                                <div className="sm:col-span-2 mt-6 mb-4">
                                    <label htmlFor="company" className="block text-sm font-semibold leading-6 text-gray-900">
                                        Company Name
                                    </label>
                                    <div className="mt-1.5">
                                        <input
                                        type="text"
                                        id="company"
                                        placeholder="Freddie's Films"

                                        name="company"
                                        autoComplete="given-name"
                                        value={values.company}

                                        onChange={handleInputChange}
                                        className={`block w-full ${values.company ? "bg-blue-100/70 focus:bg-gray-100" : "bg-gray-100"} rounded-md px-3.5 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-800/60 sm:text-sm sm:leading-6`}
                                        />
                                    </div>
                                </div>
                                <div className="sm:col-span-2">
                                    <label htmlFor="company-website" className="block text-sm font-semibold leading-6 text-gray-900">
                                        Company Website
                                    </label>
                                    <div className="relative mt-1.5">
                                            <span className='absolute inset-y-0 left-0 flex items-center rounded-md pl-3.5 text-left text-gray-900 sm:text-sm'>
                                                http://
                                            </span>
                                            <input
                                                type="text"
                                                id="company_website"

                                                name="company_website"
                                                placeholder="example.com"
                                                value={values.company_website}
                                                
                                                onChange={handleInputChange}
                                                className={`block w-full pl-16 ${values.company_website ? "bg-blue-100/70 focus:bg-gray-100" : "bg-gray-100"} rounded-md px-3.5 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-800/60 sm:text-sm sm:leading-6`}
                                            />
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                    
                    <div className="sm:col-span-2">
                        <div className="flex justify-between text-sm leading-6">
                            <label htmlFor="message" className="block text-sm font-semibold leading-6 text-gray-900">
                            How can we help you?
                            </label>
                            <p id="message-description" className="text-gray-400">
                            Max 500 characters
                            </p>
                        </div>
                        <div className="mt-1.5">
                            <textarea
                            type="message"
                            id="message"
                            placeholder='Tell us about your project and needs.'

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
                    <fieldset className="sm:col-span-2">
                        <legend className="block text-sm font-semibold leading-6 text-gray-900">How many agents are you interested in?</legend>
                        <div className="mt-4 space-y-4 text-sm leading-6 text-gray-600">
                        <div className="flex gap-x-2.5">
                            <input
                            id="agents-less-than-5"
                            name="agents"
                            defaultValue="less-than-5"
                            type="radio"
                            className="mt-1 h-4 w-4 border-gray-300 text-indigo-600 shadow-sm focus:ring-indigo-600"
                            />
                            <label htmlFor="budget-under-25k">Less than 5 agents</label>
                        </div>
                        <div className="flex gap-x-2.5">
                            <input
                            id="agents-5-10"
                            name="agents"
                            defaultValue="5-10"
                            type="radio"
                            className="mt-1 h-4 w-4 border-gray-300 text-indigo-600 shadow-sm focus:ring-indigo-600"
                            />
                            <label htmlFor="budget-25k-50k">5 - 10 agents</label>
                        </div>
                        <div className="flex gap-x-2.5">
                            <input
                            id="agents-10-50"
                            name="agents"
                            defaultValue="10-50"
                            type="radio"
                            className="mt-1 h-4 w-4 border-gray-300 text-indigo-600 shadow-sm focus:ring-indigo-600"
                            />
                            <label htmlFor="budget-50k-100k">10 – 50 agents</label>
                        </div>
                        <div className="flex gap-x-2.5">
                            <input
                            id="agents-over-50"
                            name="agents"
                            defaultValue="50"
                            type="radio"
                            className="mt-1 h-4 w-4 border-gray-300 text-indigo-600 shadow-sm focus:ring-indigo-600"
                            />
                            <label htmlFor="budget-over-100k">50+ agents</label>
                        </div>
                        </div>
                    </fieldset>
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

