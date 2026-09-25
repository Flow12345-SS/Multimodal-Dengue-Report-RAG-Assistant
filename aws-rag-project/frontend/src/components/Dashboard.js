import React, { useState } from 'react';

const API_URL = "https://your-api-gateway-id.execute-api.us-east-1.amazonaws.com/prod";

const Dashboard = () => {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");

  const handleFileChange = (e) => setFile(e.target.files[0]);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const reader = new FileReader();
    reader.onload = async () => {
      const base64 = reader.result.split(',')[1];
      try {
        const response = await fetch(`${API_URL}/upload`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: file.name, file: base64 })
        });
        const data = await response.json();
        setUploadStatus(`Success: ${data.message}`);
      } catch (err) {
        setUploadStatus("Upload failed.");
      }
      setLoading(false);
    };
    reader.readAsDataURL(file);
  };

  const handleAsk = async () => {
    if (!question.trim()) return;
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });
      const data = await response.json();
      setChatHistory([...chatHistory, { q: question, a: data.answer, evidence: data.evidence }]);
      setQuestion("");
    } catch (err) {
      alert("Error querying Bedrock.");
    }
    setLoading(false);
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '20px' }}>
      {/* Upload Section */}
      <div style={{ padding: '20px', border: '1px solid #ccc', borderRadius: '8px' }}>
        <h2>Upload Report</h2>
        <input type="file" onChange={handleFileChange} accept=".pdf,.jpg,.png" />
        <button onClick={handleUpload} disabled={!file || loading} style={btnStyle}>
          {loading ? "Uploading..." : "Upload to S3"}
        </button>
        {uploadStatus && <p style={{ color: 'green', marginTop: '10px' }}>{uploadStatus}</p>}
      </div>

      {/* Chat Section */}
      <div style={{ padding: '20px', border: '1px solid #ccc', borderRadius: '8px', display: 'flex', flexDirection: 'column' }}>
        <h2>Ask Dengue Assistant</h2>
        <div style={{ flex: 1, overflowY: 'auto', marginBottom: '20px', maxHeight: '400px' }}>
          {chatHistory.map((chat, idx) => (
            <div key={idx} style={{ marginBottom: '20px' }}>
              <p><strong>You:</strong> {chat.q}</p>
              <div style={{ background: '#f8f9fa', padding: '10px', borderRadius: '4px' }}>
                <p><strong>Assistant:</strong> {chat.a}</p>
                {chat.evidence && chat.evidence.length > 0 && (
                  <details style={{ marginTop: '10px', fontSize: '0.9em', color: '#555' }}>
                    <summary style={{ cursor: 'pointer', fontWeight: 'bold' }}>View Evidence (S3 Sources)</summary>
                    {chat.evidence.map((ev, i) => (
                      <div key={i} style={{ marginTop: '5px', padding: '5px', borderLeft: '3px solid #007bff' }}>
                        <p style={{ margin: 0 }}><em>Source: {ev.source}</em></p>
                        <p style={{ margin: '5px 0' }}>{ev.content}</p>
                      </div>
                    ))}
                  </details>
                )}
              </div>
            </div>
          ))}
        </div>
        
        <div style={{ display: 'flex', gap: '10px' }}>
          <input 
            type="text" 
            value={question} 
            onChange={(e) => setQuestion(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleAsk()}
            placeholder="Ask about platelet count, symptoms, etc..."
            style={{ flex: 1, padding: '10px', borderRadius: '4px', border: '1px solid #ccc' }}
          />
          <button onClick={handleAsk} disabled={!question || loading} style={btnStyle}>
            {loading ? "Thinking..." : "Ask"}
          </button>
        </div>
      </div>
    </div>
  );
};

const btnStyle = {
  background: '#28a745', color: 'white', border: 'none', padding: '10px 15px', borderRadius: '4px', cursor: 'pointer', marginTop: '10px', width: '100%'
};

export default Dashboard;
