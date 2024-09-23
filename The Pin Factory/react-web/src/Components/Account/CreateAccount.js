import React, { useState, Fragment } from "react";

import { Listbox, Transition } from "@headlessui/react";
import {
	ChevronUpDownIcon,
	ExclamationCircleIcon,
	ArrowLeftIcon,
} from "@heroicons/react/20/solid";
import { Link } from "react-router-dom";
import { Switch } from "@headlessui/react";
import "../Brochureware/HomeComponents/HomePage.css";

import AgentRunAcrossController from "../Brochureware/AvatarAnimation/AgentRunAcrossController";
import PhoneGeographies from "../../assets/geo/dic.js";
import LoadingCircle from "../utils/LoadingCircle";

import { createUserWithEmailAndPassword } from "firebase/auth";
import { useNavigate } from "react-router-dom";
import { auth } from "../../firebase.js";
import { functions, httpsCallable } from "../../firebase";
import "../utils/scrollbar.css";

export default function CreateAccount() {
	const [values, setValues] = useState({});
	const [selectedPhoneCountry, setSelectedPhoneCountry] = useState(
		Object.values(PhoneGeographies)[0],
	);
	const [formValid, setFormValid] = useState(false);
	const [formSubmitted, setFormSubmitted] = useState(false);
	const [submittedSuccessfully, setSubmittedSuccessfully] = useState(false);
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState(null);
	const [isCompany, setIsCompany] = useState(false);
	const navigate = useNavigate();

	const checkFormValidity = () => {
		// Check if the required fields are filled
		const requiredFields = ["first_name", "last_name", "email"];
		const isFormValid = requiredFields.every((field) => !!values[field]);
		setFormValid(isFormValid);
	};

	const handleInputChange = (event) => {
		const { name, value } = event.target;
		setValues((prevValues) => ({ ...prevValues, [name]: value }));
		checkFormValidity();
	};

	const handleCreateAccount = async (event) => {
		event.preventDefault();
		setLoading(true);
		const createUserInFirestore = httpsCallable(
			functions,
			"createuserinfirestore",
		);

		try {
			const response = await createUserWithEmailAndPassword(
				auth,
				values.email,
				values.password,
			);

			// After successful creation, add user data to Firestore
			const userDocument = {
				avatar: "https://firebasestorage.googleapis.com/v0/b/thepinfactory-42d2a.appspot.com/o/avatars%2FScreenshot%202023-04-06%20at%2002.50.55.png?alt=media&token=2c2d7a86-9075-43c2-bfee-e48b9a8f6f0e", // You can update this later, or use a default avatar
				dateOfBirth: "", // You can update this later
				email: values.email,
				firstName: values.first_name,
				userId: auth.currentUser.uid,
				lastActive: new Date().toISOString(), // Set to current time
				lastLogin: new Date().toISOString(), // Set to current time
				lastName: values.last_name,
				registrationDate: new Date().toISOString(), // Set to current time
				threadsOfUser: 0, // Default value
				username: "", // You can update this later
			};
			try {
				console.log("run createUserInFirestore");
				await createUserInFirestore({ userDocument: userDocument });
			} catch (error) {
				console.log("Error creating user in Firestore");
				console.log(error);
			}
			console.log("user document", userDocument);

			// clear inputs and error on successful creation
			console.log("Account creation successful");
			console.log(response.user.uid);
			setValues({});
			setError(null);
			// Navigate to dashboard
			navigate("/threads");
		} catch (error) {
			console.log("Account creation failed");
			console.log(error);

			switch (error.code) {
				case "auth/email-already-in-use":
					setError("This email is already in use.");
					break;
				case "auth/invalid-email":
					setError("Invalid email format.");
					break;
				case "auth/operation-not-allowed":
					setError("Account creation is not allowed.");
					break;
				case "auth/weak-password":
					setError("The password is too weak.");
					break;
				default:
					setError("Account creation failed.");
			}
		} finally {
			setLoading(false);
		}
	};

	const [search, setSearch] = useState("");
	const [phoneIsFocused, setPhoneIsFocused] = useState(false);

	const handleFocus = () => {
		setPhoneIsFocused(true);
	};

	const handleBlur = () => {
		setPhoneIsFocused(false);
	};

	return (
		<div className="fixed top-0 left-0 right-0 bottom-0">
			{/* LEFT SIDE OF CONTENT */}
			<form
				onSubmit={handleCreateAccount}
				className="absolute bg-white z-10 top-0 left-0 w-1/2 h-screen overscroll-y-none overflow-y-scroll custom-scrollbar">
				<div className="px-24 pt-12">
					<button onClick={() => navigate("/")}>
						<ArrowLeftIcon className="w-6 text-gray-900 hover:opacity-70 mb-4" />
					</button>

					<div className="mb-8">
						<h2 className="text-3xl font-bold tracking-tight text-gray-900">
							Create account
						</h2>
						<p className="mt-2 text-md leading-6 text-gray-600">
							Create your account here to start making agents.
						</p>
					</div>

					<div className="grid grid-cols-1 gap-x-8 gap-y-6 sm:grid-cols-2">
						{/* START FIRST NAME */}
						<div>
							<label
								htmlFor="first-name"
								className="block text-sm font-semibold leading-6 text-gray-900">
								First name
							</label>
							<div className="mt-1.5">
								<input
									type="text"
									id="first_name"
									placeholder="Freddie"
									name="first_name"
									autoComplete="given-name"
									value={values.firstName}
									onChange={handleInputChange}
									className={`block w-full ${
										values.first_name
											? "bg-blue-100/70 focus:bg-gray-100"
											: "bg-gray-100"
									} rounded-md px-3.5 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-800/60 sm:text-sm sm:leading-6`}
								/>
							</div>
						</div>
						{/* END FIRST NAME */}

						{/* START LAST NAME */}
						<div>
							<label
								htmlFor="last-name"
								className="block text-sm font-semibold leading-6 text-gray-900">
								Last name
							</label>
							<div className="mt-1.5">
								<input
									type="text"
									id="last_name"
									placeholder="Frank"
									name="last_name"
									autoComplete="given-name"
									value={values.last_name}
									onChange={handleInputChange}
									className={`block w-full ${
										values.last_name
											? "bg-blue-100/70 focus:bg-gray-100"
											: "bg-gray-100"
									} rounded-md px-3.5 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-800/60 sm:text-sm sm:leading-6`}
								/>
							</div>
						</div>
						{/* END LAST NAME */}

						{/* START PHONE */}
						<div className="sm:col-span-2">
							<label
								htmlFor="phone"
								className="block text-sm font-semibold leading-6 text-gray-900">
								Phone
							</label>

							<div className="relative mt-1.5">
								{/* PHONE COUNTRY CODE AND DROP DOWN */}
								<Listbox
									value={selectedPhoneCountry}
									onChange={setSelectedPhoneCountry}>
									{({ open }) => (
										<>
											<div
												className={`absolute inset-y-0 left-0 flex items-center rounded-md ${
													open
														? "ring-2 ring-blue-800/60"
														: ""
												}`}>
												<div className="relative">
													<Listbox.Button
														className={`relative mt-1.5 pl-3.5 pr-1.5 cursor-default text-left text-gray-900 sm:text-sm sm:leading-6`}>
														<div className="flex items-center justify-between">
															<div className="flex items-center justify-start space-x-1">
																<selectedPhoneCountry.flag />
																<ChevronUpDownIcon
																	className="h-4 w-4 text-gray-900"
																	aria-hidden="true"
																/>
															</div>
														</div>
													</Listbox.Button>

													<Transition
														show={open}
														as={Fragment}
														leave="transition ease-in duration-100"
														leaveFrom="opacity-100"
														leaveTo="opacity-0">
														<Listbox.Options className="absolute z-10 mt-4 w-72 rounded-md bg-white py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
															{/* Search field to filter PhoneGeographies map */}
															<div className="flex items-center pt-1 pb-2 mb-1 px-3 border-b border-gray-900/10">
																<input
																	type="text"
																	name="search"
																	id="search"
																	placeholder="Search"
																	value={
																		search
																	}
																	onChange={(
																		e,
																	) =>
																		setSearch(
																			e
																				.target
																				.value,
																		)
																	}
																	onKeyDown={(
																		e,
																	) => {
																		if (
																			e.key ===
																			" "
																		) {
																			e.preventDefault();
																			setSearch(
																				(
																					prevSearch,
																				) =>
																					prevSearch +
																					" ",
																			);
																		}
																	}}
																	className="ml-2 block w-full text-gray-900 placeholder-gray-400 outline-none ring-0 text-sm"
																/>
															</div>

															<div className="max-h-48 overflow-auto custom-scrollbar">
																{// If search field is not empty, filter PhoneGeographies map by search field
																(search
																	? Object.values(
																			PhoneGeographies,
																	  ).filter(
																			(
																				geo,
																			) =>
																				geo.name
																					.toLowerCase()
																					.includes(
																						search.toLowerCase(),
																					) ||
																				geo.code
																					.toLowerCase()
																					.includes(
																						search.toLowerCase(),
																					),
																	  )
																	: Object.values(
																			PhoneGeographies,
																	  )
																).map((geo) => (
																	<Listbox.Option
																		key={
																			geo.code
																		}
																		className={({
																			active,
																		}) =>
																			classNames(
																				active
																					? "rounded-md bg-gray-100"
																					: "text-gray-900",
																				"relative cursor-default select-none mx-1 py-2",
																			)
																		}
																		value={
																			geo
																		}>
																		{({
																			selected,
																		}) => (
																			<>
																				<div className="flex justify-between mx-4">
																					<div className="flex items-center">
																						{/* ADD FLAG HERE */}
																						<div className="mr-4">
																							<geo.flag />
																						</div>

																						<span
																							className={classNames(
																								selected
																									? "font-semibold"
																									: "font-normal",
																								"block truncate",
																							)}>
																							{
																								geo.name
																							}
																						</span>
																					</div>
																					<span
																						className={
																							"font-normal block truncate text-gray-500"
																						}>
																						+
																						{
																							geo.phoneNumberPrefix
																						}
																					</span>
																				</div>
																			</>
																		)}
																	</Listbox.Option>
																))}
															</div>
														</Listbox.Options>
													</Transition>
												</div>
											</div>
										</>
									)}
								</Listbox>

								{/* ACTUAL PHONE ENTRY */}
								<div
									className={`block w-full pl-16 flex items-center space-x-4 ${
										values.phone
											? "bg-blue-100/70 focus:bg-gray-100"
											: "bg-gray-100"
									} rounded-md px-3.5 py-2 text-gray-500 ${
										phoneIsFocused
											? "ring-2 ring-blue-800/60"
											: ""
									} sm:text-sm sm:leading-6`}>
									<p className="">
										{"+" +
											selectedPhoneCountry.phoneNumberPrefix}
									</p>
									<input
										type="tel"
										id="phone"
										placeholder={
											selectedPhoneCountry.examplePhoneNumber
										}
										name="phone"
										autoComplete="given-name"
										value={values.phone}
										onChange={handleInputChange}
										onFocus={() => setPhoneIsFocused(true)}
										onBlur={() => setPhoneIsFocused(false)}
										className={`w-full bg-transparent outline-none text-gray-900 placeholder:text-gray-400`}
									/>
								</div>
							</div>
						</div>
						{/* END PHONE */}

						{/* START EMAIL */}
						<div className="sm:col-span-2">
							<label
								htmlFor="email"
								className="block text-sm font-semibold leading-6 text-gray-900">
								Email
							</label>
							<div className="mt-1.5">
								<input
									type="text"
									id="email"
									placeholder="freddie@example.com"
									name="email"
									autoComplete="given-name"
									value={values.email}
									onChange={handleInputChange}
									className={`block w-full ${
										values.email
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
									type="password"
									id="password"
									placeholder="Create password"
									name="password"
									value={values.password}
									onChange={handleInputChange}
									className={`block w-full ${
										values.password
											? "bg-blue-100/70 focus:bg-gray-100"
											: "bg-gray-100"
									} rounded-md px-3.5 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-800/60 sm:text-sm sm:leading-6`}
								/>
							</div>
						</div>
						{/* END PASSWORD */}
					</div>
					{error && (
						<div className="flex space-x-2 items-center justify-start mt-6">
							<ExclamationCircleIcon className="h-4 w-4 text-red-700" />
							<p className="text-start font-medium text-xs text-red-700">
								{error}
							</p>
						</div>
					)}
					<div className="mt-10">
						<button
							type="submit"
							disabled={loading}
							className="block w-full rounded-md bg-green-700 px-3.5 py-2.5 text-center text-sm font-semibold text-white shadow-sm hover:bg-green-600 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600">
							<div className="flex justify-between items-center">
								<div className="w-8" />
								Create account
								<div className="w-8">
									{loading && (
										<LoadingCircle size="5" color="white" />
									)}
								</div>
							</div>
						</button>
					</div>
					<p className="mt-4 text-sm leading-6 text-gray-500">
						By creating an account, I agree to the{" "}
						<a
							href="#"
							className="font-semibold text-green-900 hover:text-green-600">
							privacy&nbsp;policy
						</a>{" "}
						and the{" "}
						<a
							href="#"
							className="font-semibold text-green-900 hover:text-green-600">
							terms&nbsp;of&nbsp;service
						</a>
						.
					</p>
				</div>
			</form>

			{/* RIGHT SIDE */}
			<div className="absolute z-0 top-0 right-0 h-screen w-1/2 backdrop shadow-[inset_0_0px_50px_20px_#9ca3af]">
				{/* BACKGROUND */}
				<svg
					className="absolute inset-0 h-full w-full stroke-blue-900/30 [mask-image:radial-gradient(100%_100%_at_bottom_right,#FFFFFF80,#FFFFFF80)]"
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
						className="overflow-visible fill-gray-100">
						<path d="M-470.5 0h201v201h-201Z" strokeWidth={0} />
					</svg>
					<svg
						x="100%"
						y={-1}
						className="overflow-visible fill-gray-100">
						<path d="M-470.5 0h201v201h-201Z" strokeWidth={0} />
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
	);
}

function classNames(...classes) {
	return classes.filter(Boolean).join(" ");
}
