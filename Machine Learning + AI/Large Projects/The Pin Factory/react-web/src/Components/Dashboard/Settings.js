import { Fragment, useState, useEffect } from "react";
import { Dialog, Transition } from "@headlessui/react";
import {
	ChartBarSquareIcon,
	Cog6ToothIcon,
	FolderIcon,
	GlobeAltIcon,
	ServerIcon,
	SignalIcon,
	XMarkIcon,
} from "@heroicons/react/24/outline";
import { Bars3Icon, MagnifyingGlassIcon } from "@heroicons/react/20/solid";
import DashboardHeader from "./DashboardHeader";
import { db } from "../../firebase";
import { updateDoc, doc, getDoc } from "firebase/firestore";
import {
	getStorage,
	ref,
	uploadBytesResumable,
	getDownloadURL,
} from "firebase/storage";

const navigation = [
	{ name: "Projects", href: "#", icon: FolderIcon, current: false },
	{ name: "Deployments", href: "#", icon: ServerIcon, current: false },
	{ name: "Activity", href: "#", icon: SignalIcon, current: false },
	{ name: "Domains", href: "#", icon: GlobeAltIcon, current: false },
	{ name: "Usage", href: "#", icon: ChartBarSquareIcon, current: false },
	{ name: "Settings", href: "#", icon: Cog6ToothIcon, current: true },
];
const teams = [
	{ id: 1, name: "Planetaria", href: "#", initial: "P", current: false },
	{ id: 2, name: "Protocol", href: "#", initial: "P", current: false },
	{ id: 3, name: "Tailwind Labs", href: "#", initial: "T", current: false },
];
const secondaryNavigation = [
	{ name: "Account", href: "#", current: true },
	{ name: "Notifications", href: "#", current: false },
	{ name: "Billing", href: "#", current: false },
	{ name: "Teams", href: "#", current: false },
	{ name: "Integrations", href: "#", current: false },
];

function classNames(...classes) {
	return classes.filter(Boolean).join(" ");
}

export default function Settings() {
	const [sidebarOpen, setSidebarOpen] = useState(false);
	const [name, setName] = useState("");
	const [username, setUsername] = useState("");
	const [emailAddress, setEmailAddress] = useState("");
	const [avatar, setAvatar] = useState(
		"https://cdn-icons-png.flaticon.com/512/2815/2815428.png",
	);

	const [showSuccessMessage, setShowSuccessMessage] = useState(false);

	const handleChangeAvatar = (event) => {
		// const file = event.target.files[0];
		// const storage = getStorage();
		// const storageRef = ref(storage, "avatars/" + file.name);
		// const uploadTask = uploadBytesResumable(storageRef, file);
		// // Listen for state changes, errors, and completion of the upload.
		// uploadTask.on(
		// 	"state_changed",
		// 	(snapshot) => {
		// 		// Get task progress, including the number of bytes uploaded and the total number of bytes to be uploaded
		// 		var progress =
		// 			(snapshot.bytesTransferred / snapshot.totalBytes) * 100;
		// 		console.log("Upload is " + progress + "% done");
		// 	},
		// 	(error) => {
		// 		// Handle unsuccessful uploads
		// 		console.error(error);
		// 	},
		// 	() => {
		// 		// Handle successful uploads on complete
		// 		getDownloadURL(uploadTask.snapshot.ref).then((downloadURL) => {
		// 			console.log("File available at", downloadURL);
		// 			setAvatar(downloadURL); // Set avatar state to the download URL of the uploaded image
		// 		});
		// 	},
		// );
	};

	useEffect(() => {
		// const fetchUserData = async () => {
		// 	const docRef = doc(db, "users", "VQgNYRdfgFlcczxpDbHP");
		// 	const docSnap = await getDoc(docRef);
		// 	if (docSnap.exists()) {
		// 		const data = docSnap.data();
		// 		setName(data.firstName);
		// 		setUsername(data.username);
		// 		setEmailAddress(data.email);
		// 		setAvatar(data.avatar);
		// 		console.log("the data is", data);
		// 		// setAvatar(data.avatar);
		// 	} else {
		// 		console.log("No such document!");
		// 	}
		// };
		// fetchUserData();
	}, []);
	const updateUserData = async (e) => {
		// e.preventDefault();
		// const docRef = doc(db, "users", "VQgNYRdfgFlcczxpDbHP");
		// const updateData = {
		// 	firstName: name,
		// 	username: username,
		// 	email: emailAddress,
		// 	avatar: avatar,
		// };
		// try {
		// 	await updateDoc(docRef, updateData);
		// 	setShowSuccessMessage(true);
		// 	setTimeout(() => setShowSuccessMessage(false), 5000);
		// } catch (e) {
		// 	console.log(e);
		// }
	};

	return (
		<>
			{/*
        This example requires updating your template:

        ```
        <html class="h-full bg-gray-900">
        <body class="h-full">
        ```
      */}
			<div>
				<DashboardHeader HeaderHeight={3.5} />

				<div className="">
					<main>
						<header className="border-b border-gray-200 ">
							{/* Secondary navigation */}
							<nav className="flex overflow-x-auto py-4 mt-[3.5rem]">
								<ul
									role="list"
									className="flex min-w-full flex-none gap-x-6 px-4 text-sm font-semibold leading-6 text-gray-400 sm:px-6 lg:px-8">
									{secondaryNavigation.map((item) => (
										<li key={item.name}>
											<a
												href={item.href}
												className={
													item.current
														? "text-indigo-600"
														: ""
												}>
												{item.name}
											</a>
										</li>
									))}
								</ul>
							</nav>
						</header>

						{/* Settings forms */}
						<div className="divide-y divide-white/5">
							<div className="grid max-w-7xl grid-cols-1 gap-x-8 gap-y-10 px-4 py-16 sm:px-6 md:grid-cols-3 lg:px-8">
								<div>
									<h2 className="text-base font-semibold leading-7 text-gray-800">
										Personal Information
									</h2>
									<p className="mt-1 text-sm leading-6 text-gray-400">
										Set some basic information about
										yourself.
									</p>
								</div>

								<form
									className="md:col-span-2"
									onSubmit={updateUserData}>
									<div className="grid grid-cols-1 gap-x-6 gap-y-8 sm:max-w-xl sm:grid-cols-6">
										<div className="col-span-full flex items-center gap-x-8">
											<img
												src={avatar}
												alt=""
												className="h-24 w-24 flex-none rounded-lg bg-gray-800 object-cover"
											/>
											<div>
												<label className="rounded-md bg-white/10 px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm hover:bg-white/20 cursor-pointer">
													Change avatar
													<input
														type="file"
														accept="image/*"
														onChange={
															handleChangeAvatar
														}
														className="hidden"
													/>
												</label>
												<p className="mt-2 text-xs leading-5 text-gray-900">
													JPG, GIF or PNG. 1MB max.
												</p>
											</div>
										</div>

										<div className="col-span-full">
											<label
												htmlFor="fullname"
												className="block text-sm font-medium leading-6 text-gray-800">
												Full Name
											</label>
											<div className="mt-2">
												<input
													id="fullname"
													name="fullname"
													value={name}
													onChange={(e) =>
														setName(e.target.value)
													}
													autoComplete="given-name"
													type="text"
													className="block w-full rounded-md border-gray-300 bg-white/5 py-1.5 text-gray-800 shadow-sm ring-1 ring-inset ring-white/10 focus:ring-2 focus:ring-inset focus:ring-indigo-500 sm:text-sm sm:leading-6"
												/>
											</div>
										</div>

										<div className="col-span-full">
											<label
												htmlFor="email"
												className="block text-sm font-medium leading-6 text-gray-800">
												Email address
											</label>
											<div className="mt-2">
												<input
													id="email"
													name="email"
													value={emailAddress}
													onChange={(e) =>
														setEmailAddress(
															e.target.value,
														)
													}
													type="email"
													autoComplete="email"
													className="block w-full rounded-md border-gray-300 bg-white/5 py-1.5 text-gray-800 shadow-sm ring-1 ring-inset ring-white/10 focus:ring-2 focus:ring-inset focus:ring-indigo-500 sm:text-sm sm:leading-6"
												/>
											</div>
										</div>

										<div className="col-span-full">
											<label
												htmlFor="username"
												className="block text-sm font-medium leading-6 text-gray-800">
												Username
											</label>
											<div className="mt-2">
												<input
													id="username"
													name="username"
													value={username}
													onChange={(e) =>
														setUsername(
															e.target.value,
														)
													}
													autoComplete="email"
													type="text"
													className="block w-full rounded-md border-gray-300 bg-white/5 py-1.5 text-gray-800 shadow-sm ring-1 ring-inset ring-white/10 focus:ring-2 focus:ring-inset focus:ring-indigo-500 sm:text-sm sm:leading-6"
												/>
											</div>
										</div>

										<div className="col-span-full">
											<label
												htmlFor="timezone"
												className="block text-sm font-medium leading-6 text-gray-800">
												Timezone
											</label>
											<div className="mt-2">
												<select
													id="timezone"
													name="timezone"
													className="block w-full rounded-md border-gray-300 bg-white/5 py-1.5 text-gray-800 shadow-sm ring-1 ring-inset ring-white/10 focus:ring-2 focus:ring-inset focus:ring-indigo-500 sm:text-sm sm:leading-6 [&_*]:text-black">
													<option>
														Pacific Standard Time
													</option>
													<option>
														Eastern Standard Time
													</option>
													<option>
														Greenwich Mean Time
													</option>
												</select>
											</div>
										</div>
									</div>

									<div className="mt-8 flex">
										<button
											onClick={updateUserData}
											type="submit"
											className="rounded bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600">
											Save
										</button>
										{showSuccessMessage && (
											<p className="text-gray-800 pt-1 px-5">
												Your information was updated
												successfully!
											</p>
										)}
									</div>
								</form>
							</div>

							<div className="grid max-w-7xl grid-cols-1 gap-x-8 gap-y-10 px-4 py-16 sm:px-6 md:grid-cols-3 lg:px-8">
								<div>
									<h2 className="text-base font-semibold leading-7 text-gray-800">
										Change password
									</h2>
									<p className="mt-1 text-sm leading-6 text-gray-400">
										Update your password associated with
										your account.
									</p>
								</div>

								<form className="md:col-span-2">
									<div className="grid grid-cols-1 gap-x-6 gap-y-8 sm:max-w-xl sm:grid-cols-6">
										<div className="col-span-full">
											<label
												htmlFor="current-password"
												className="block text-sm font-medium leading-6 text-gray-800">
												Current password
											</label>
											<div className="mt-2">
												<input
													id="current-password"
													name="current_password"
													type="password"
													autoComplete="current-password"
													className="block w-full rounded-md border-gray-300 bg-white/5 py-1.5 text-gray-800 shadow-sm ring-1 ring-inset ring-white/10 focus:ring-2 focus:ring-inset focus:ring-indigo-500 sm:text-sm sm:leading-6"
												/>
											</div>
										</div>

										<div className="col-span-full">
											<label
												htmlFor="new-password"
												className="block text-sm font-medium leading-6 text-gray-800">
												New password
											</label>
											<div className="mt-2">
												<input
													id="new-password"
													name="new_password"
													type="password"
													autoComplete="new-password"
													className="block w-full rounded-md border-gray-300 bg-white/5 py-1.5 text-gray-800 shadow-sm ring-1 ring-inset ring-white/10 focus:ring-2 focus:ring-inset focus:ring-indigo-500 sm:text-sm sm:leading-6"
												/>
											</div>
										</div>

										<div className="col-span-full">
											<label
												htmlFor="confirm-password"
												className="block text-sm font-medium leading-6 text-gray-800">
												Confirm password
											</label>
											<div className="mt-2">
												<input
													id="confirm-password"
													name="confirm_password"
													type="password"
													autoComplete="new-password"
													className="block w-full rounded-md border-gray-300 bg-white/5 py-1.5 text-gray-800 shadow-sm ring-1 ring-inset ring-white/10 focus:ring-2 focus:ring-inset focus:ring-indigo-500 sm:text-sm sm:leading-6"
												/>
											</div>
										</div>
									</div>

									<div className="mt-8 flex">
										<button
											type="submit"
											className="rounded bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600">
											Save
										</button>
									</div>
								</form>
							</div>

							<div className="grid max-w-7xl grid-cols-1 gap-x-8 gap-y-10 px-4 py-16 sm:px-6 md:grid-cols-3 lg:px-8">
								<div>
									<h2 className="text-base font-semibold leading-7 text-gray-800">
										Log out other sessions
									</h2>
									<p className="mt-1 text-sm leading-6 text-gray-400">
										Please enter your password to confirm
										you would like to log out of your other
										sessions across all of your devices.
									</p>
								</div>

								<form className="md:col-span-2">
									<div className="grid grid-cols-1 gap-x-6 gap-y-8 sm:max-w-xl sm:grid-cols-6">
										<div className="col-span-full">
											<label
												htmlFor="logout-password"
												className="block text-sm font-medium leading-6 text-gray-800">
												Your password
											</label>
											<div className="mt-2">
												<input
													id="logout-password"
													name="password"
													type="password"
													autoComplete="current-password"
													className="block w-full rounded-md border-gray-300 bg-white/5 py-1.5 text-gray-800 shadow-sm ring-1 ring-inset ring-white/10 focus:ring-2 focus:ring-inset focus:ring-indigo-500 sm:text-sm sm:leading-6"
												/>
											</div>
										</div>
									</div>

									<div className="mt-8 flex">
										<button
											type="submit"
											className="rounded bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600">
											Log out other sessions
										</button>
									</div>
								</form>
							</div>

							<div className="grid max-w-7xl grid-cols-1 gap-x-8 gap-y-10 px-4 py-16 sm:px-6 md:grid-cols-3 lg:px-8">
								<div>
									<h2 className="text-base font-semibold leading-7 text-gray-800">
										Delete account
									</h2>
									<p className="mt-1 text-sm leading-6 text-gray-400">
										No longer want to use our service? You
										can delete your account here. This
										action is not reversible. All
										information related to this account will
										be deleted permanently.
									</p>
								</div>

								<form className="flex items-start md:col-span-2">
									<button
										type="submit"
										className="rounded-md bg-red-500 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-red-400">
										Yes, delete my account
									</button>
								</form>
							</div>
						</div>
					</main>
				</div>
			</div>
		</>
	);
}
