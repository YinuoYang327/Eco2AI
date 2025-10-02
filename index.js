import { useEffect, useState } from "react";

export default function Home() {
  const [data, setData] = useState([]);

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch(process.env.NEXT_PUBLIC_JSONBIN_URL);
        const json = await res.json();
        setData(json);
      } catch (err) {
        console.error("Error fetching data:", err);
      }
    }
    fetchData();
  }, []);

  return (
    <div style={{ padding: "2rem" }}>
      <h1>🌍 Eco2AI Dashboard</h1>
      <table border="1" cellPadding="5">
        <thead>
          <tr>
            <th>Project</th>
            <th>Experiment</th>
            <th>Start Time</th>
            <th>Duration (s)</th>
            <th>CO2 (kg)</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => (
            <tr key={idx}>
              <td>{row.project_name}</td>
              <td>{row.experiment_description}</td>
              <td>{row.start_time}</td>
              <td>{row["duration(s)"]}</td>
              <td>{row["CO2_emissions(kg)"]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
