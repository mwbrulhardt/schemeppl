'use client';
import { useEffect, useState } from "react";
import init, { add } from "../pkg/wasm";

export default function Home() {
  const [result, setResult] = useState<number | null>(null);
  const [num1, setNum1] = useState<string>("0");
  const [num2, setNum2] = useState<string>("0");
  const [isWasmReady, setIsWasmReady] = useState(false);

  useEffect(() => {
    init().then(() => {
      setIsWasmReady(true);
    });
  }, []);

  const handleCalculate = () => {
    if (isWasmReady) {
      setResult(add(Number(num1), Number(num2)));
    }
  };

  return (
    <div className="p-4">
      <div className="flex gap-4 items-center">
        <input
          type="number"
          value={num1}
          onChange={(e) => setNum1(e.target.value)}
          className="border p-2 rounded"
        />
        <span>+</span>
        <input
          type="number"
          value={num2}
          onChange={(e) => setNum2(e.target.value)}
          className="border p-2 rounded"
        />
        <button
          onClick={handleCalculate}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
          disabled={!isWasmReady}
        >
          Calculate
        </button>
      </div>
      {result !== null && (
        <div className="mt-4">
          Result: {result}
        </div>
      )}
    </div>
  );
}
