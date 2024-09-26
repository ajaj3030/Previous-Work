import React from "react";
import "./utils.css"

const LoadingCircle = ({size, color}) => (
    <div className={`animate-spin loader ease-linear rounded-full border-[3px] border-t-4 border-${color} h-${size} w-${size}`}></div>
);

export default LoadingCircle;