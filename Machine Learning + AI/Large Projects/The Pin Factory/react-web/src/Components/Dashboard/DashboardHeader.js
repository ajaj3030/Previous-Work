import React, { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { UserIcon } from "@heroicons/react/20/solid";
import { signOut } from "firebase/auth";
import { auth } from "../../firebase";
import { Fragment } from "react";
import { Menu, Transition } from "@headlessui/react";
import ProfilePicture from "../../assets/test_pp/IMG_2109.PNG";
import {
	ArchiveBoxIcon,
	ArrowRightCircleIcon,
	ChevronDownIcon,
	DocumentDuplicateIcon,
	HeartIcon,
	PencilSquareIcon,
	TrashIcon,
	UserPlusIcon,
	ArrowLeftOnRectangleIcon,
	UserCircleIcon,
	Cog8ToothIcon,
} from "@heroicons/react/20/solid";
import { functions, httpsCallable } from "../../firebase";

export default function DashboardHeader({ HeaderHeight }) {
	const location = useLocation();
	const { pathname } = location;
	const navigation = [
		{ name: "Threads", to: "/threads", current: pathname === "/threads" },
		{ name: "Agents", to: "/agents", current: pathname === "/agents" },
		// { name: "Plug-Ins", to: "/plugins", current: pathname === "/plugins" },
	];
	return (
		<nav className="fixed top-0 left-0 right-0 z-10 bg-gray-50">
			<div
				className="flex items-center justify-between border-b border-gray-200 px-8 mt-0.5"
				style={{ height: `${HeaderHeight}vh` }}>
				<h1 className="text-lg font-bold text-gray-900 text-start w-48">
					the pin factory
				</h1>

				{/* <div className="flex space-x-4">
					{navigation.map((item) => (
						<Link
							to={item.to}
							className={`inline-flex items-center text-sm rounded-md font-semibold py-1 px-4 ${
								item.current
									? "bg-brand-sd text-white"
									: "bg-transparent text-gray-900 hover:bg-gray-200"
							}`}
							aria-current={item.current ? "page" : undefined}>
							{item.name}
						</Link>
					))}
				</div> */}

				{/* FAR RIGHT OF HEADER */}
				<div className="flex justify-end w-48">
					{/* <DropdownMenu /> */}
				</div>
			</div>
		</nav>
	);
}

const DropdownMenu = () => {
	const [user, setUser] = useState(null);
	const [avatar, setAvatar] = useState(
		"https://cdn1.vectorstock.com/i/1000x1000/31/95/user-sign-icon-person-symbol-human-avatar-vector-12693195.jpg",
	);
	const userId = auth.currentUser.uid; // Gets the current user's ID
	console.log("user id is", userId);
	useEffect(() => {
		const getUserDetails = httpsCallable(functions, "getUserDetails");

		getUserDetails({ userId: userId })
			.then((result) => {
				setUser(result.data);
				setAvatar(result.data.avatar);
				console.log("profile picture", result);
			})
			.catch((error) => {
				console.error("Error getting user data: ", error);
			});
	}, [userId]);
	const [loading, setLoading] = useState(false);

	const handleSignOut = async () => {
		setLoading(true);
		try {
			await signOut(auth);
			console.log("User signed out");
		} catch (error) {
			console.error("Failed to sign out user:", error);
		} finally {
			setLoading(false);
		}
	};
	function classNames(...classes) {
		return classes.filter(Boolean).join(" ");
	}

	const [isHovered, setIsHovered] = useState(false);

	return (
		<Menu
			onMouseEnter={() => setIsHovered(true)}
			onMouseLeave={() => setIsHovered(false)}
			as="div"
			className="relative inline-block text-left">
			<Menu.Button className="inline-flex cursor-default w-full justify-center gap-x-1.5 rounded-md px-3 py-2 text-sm font-semibold text-gray-900 ring-gray-300">
				<div className="w-8 h-8 relative flex-shrink-0 rounded-full outline outline-2 outline-gray-900/10">
					<img
						src={avatar}
						alt="human name"
						className="absolute top-0 left-0 w-full h-full rounded-full object-cover"
					/>
				</div>
			</Menu.Button>

			<Transition
				show={isHovered}
				as={Fragment}
				enter="transition ease-out duration-100"
				enterFrom="transform opacity-0 scale-95"
				enterTo="transform opacity-100 scale-100"
				leave="transition ease-in duration-75"
				leaveFrom="transform opacity-100 scale-100"
				leaveTo="transform opacity-0 scale-95">
				<div className="absolute -mt-1 right-0 z-10 w-56 origin-top-right divide-y divide-gray-100 rounded-md bg-white shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
					<div className="py-1">
						<Menu.Item>
							{({ active }) => (
								<Link
									to="/accounts"
									className={classNames(
										active
											? "bg-gray-100 text-gray-900"
											: "text-gray-700",
										"group flex items-center px-4 py-2 text-sm",
									)}>
									<UserCircleIcon
										className="mr-3 h-5 w-5 text-gray-400 group-hover:text-gray-500"
										aria-hidden="true"
									/>
									Accounts
								</Link>
							)}
						</Menu.Item>
						<Menu.Item>
							{({ active }) => (
								<Link
									to="/settings"
									className={classNames(
										active
											? "bg-gray-100 text-gray-900"
											: "text-gray-700",
										"group flex items-center px-4 py-2 text-sm",
									)}>
									<Cog8ToothIcon
										className="mr-3 h-5 w-5 text-gray-400 group-hover:text-gray-500"
										aria-hidden="true"
									/>
									Settings
								</Link>
							)}
						</Menu.Item>
						<Menu.Item>
							{({ active }) => (
								<button
									onClick={handleSignOut}
									disabled={loading}
									className={classNames(
										active
											? "bg-gray-100 text-gray-900"
											: "text-gray-700",
										"group flex items-center px-4 py-2 text-sm w-full",
									)}>
									<ArrowLeftOnRectangleIcon
										className="mr-3 h-5 w-5 text-gray-400 group-hover:text-gray-500"
										aria-hidden="true"
									/>

									{loading ? "Signing Out..." : "Sign Out"}
								</button>
							)}
						</Menu.Item>
					</div>
				</div>
			</Transition>
		</Menu>
	);
};
