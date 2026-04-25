import "./App.css";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import Home from "./components/Home";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="*" element={<div className="text-center pt-20">404 Page Not Found</div>} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
