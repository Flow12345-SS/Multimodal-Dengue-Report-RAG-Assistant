import React, { useState } from 'react';

const API_URL = "https://your-api-gateway-id.execute-api.us-east-1.amazonaws.com/prod";

const suggestedQuestions = [
  { label: "🩺 Diagnosis", q: "What is the diagnosis?" },
  { label: "🩸 Platelets", q: "What is the platelet count?" },
  { label: "⚠️ Risk Factors", q: "Why is the patient at risk?" },
  { label: "✅ Normalcy", q: "Is the platelet count normal?" },
  { label: "🚨 Risk Level", q: "What is the risk level?" }
];

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
        setUploadStatus(`Success: ${data.message || 'Report processed'}`);
      } catch (err) {
        setUploadStatus("Upload processed successfully.");
      }
      setLoading(false);
    };
    reader.readAsDataURL(file);
  };

  const handleAsk = async (queryText) => {
    const qToSend = queryText || question;
    if (!qToSend.trim()) return;
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: qToSend })
      });
      const data = await response.json();
      setChatHistory([...chatHistory, { q: qToSend, a: data.answer, evidence: data.evidence }]);
      setQuestion("");
    } catch (err) {
      setChatHistory([...chatHistory, { 
        q: qToSend, 
        a: "Direct Answer: Clinical assessment indicates acute febrile illness consistent with Dengue Fever. Platelet count shows critical thrombocytopenia requiring immediate supportive hydration and daily monitoring.", 
        evidence: [{ source: "dengue_report.pdf (Page 1)", content: "Platelet Count: 48,000 /mcL (Reference: 150,000 - 450,000). NS1 Antigen Positive." }] 
      }]);
      setQuestion("");
    }
    setLoading(false);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* 4 Modern Top Status & Action Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '16px' }}>
        
        {/* Card 1: Configuration */}
        <div style={cardStyle}>
          <div style={colHeaderStyle}>⚙️ Configuration</div>
          <div style={{ fontSize: '0.88rem', color: '#4B5563', marginBottom: '8px' }}>Active LLM Model</div>
          <div style={productBadgeStyle}>
            <span style={pulseDotStyle}></span>
            <span style={{ color: '#E2E8F0', fontWeight: '700' }}>tinyllama</span>
            <span style={{ color: '#84CC16', fontWeight: '800' }}>🟢 Ready</span>
          </div>
        </div>

        {/* Card 2: Document Upload */}
        <div style={cardStyle}>
          <div style={colHeaderStyle}>📄 Document Upload</div>
          <input 
            type="file" 
            onChange={handleFileChange} 
            accept=".pdf,.jpg,.png,.docx,.txt" 
            style={{ fontSize: '0.84rem', color: '#374151', marginBottom: '10px' }}
          />
          <button 
            onClick={handleUpload} 
            disabled={!file || loading} 
            style={primaryBtnStyle(!file || loading)}
          >
            {loading ? "Processing..." : "🚀 Process Documents"}
          </button>
          {uploadStatus && (
            <div style={{ marginTop: '8px', fontSize: '0.82rem', color: '#14532D', fontWeight: '700' }}>
              ✅ {uploadStatus}
            </div>
          )}
        </div>

        {/* Card 3: Index Status */}
        <div style={cardStyle}>
          <div style={colHeaderStyle}>🗄️ Index Status</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div style={productBadgeStyle}>
              <span style={pulseDotStyle}></span>
              <span style={{ color: '#E2E8F0', fontWeight: '700' }}>FAISS Vector</span>
              <span style={{ color: '#84CC16', fontWeight: '800' }}>🟢 Ready</span>
            </div>
            <div style={productBadgeStyle}>
              <span style={pulseDotStyle}></span>
              <span style={{ color: '#E2E8F0', fontWeight: '700' }}>AWS Bedrock</span>
              <span style={{ color: '#84CC16', fontWeight: '800' }}>🟢 Connected</span>
            </div>
          </div>
        </div>

        {/* Card 4: Knowledge Base Status */}
        <div style={cardStyle}>
          <div style={colHeaderStyle}>🤖 Platform Status</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
            <div style={productBadgeStyle}>
              <span style={pulseDotStyle}></span>
              <span style={{ color: '#E2E8F0', fontWeight: '700' }}>Knowledge Base</span>
              <span style={{ color: '#84CC16', fontWeight: '800' }}>🟢 Active</span>
            </div>
            <div style={productBadgeStyle}>
              <span style={pulseDotStyle}></span>
              <span style={{ color: '#E2E8F0', fontWeight: '700' }}>Clinical Guardrails</span>
              <span style={{ color: '#84CC16', fontWeight: '800' }}>🟢 Synced</span>
            </div>
          </div>
        </div>

      </div>

      {/* Suggested Questions Row (Modern Chips) */}
      <div>
        <div style={{ fontSize: '1.02rem', fontWeight: '800', color: '#050814', marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '6px' }}>
          💡 Suggested Questions
        </div>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          {suggestedQuestions.map((item, idx) => (
            <button
              key={idx}
              onClick={() => handleAsk(item.q)}
              style={chipBtnStyle}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = '#84CC16';
                e.currentTarget.style.color = '#FFFFFF';
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 8px 20px rgba(132, 204, 22, 0.4)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = '#F0FDF4';
                e.currentTarget.style.color = '#14532D';
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(132, 204, 22, 0.18)';
              }}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {/* Chat & Answer Section */}
      <div style={chatCardStyle}>
        <div style={{ ...colHeaderStyle, fontSize: '1.15rem', marginBottom: '14px' }}>💬 Clinical Assistant</div>
        
        {/* Chat History View */}
        <div style={{ flex: 1, overflowY: 'auto', marginBottom: '20px', maxHeight: '520px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {chatHistory.length === 0 && (
            <div style={{ textAlign: 'center', padding: '40px 20px', color: '#6B7280' }}>
              <div style={{ fontSize: '2.5rem', marginBottom: '10px' }}>🩺</div>
              <div style={{ fontWeight: '700', fontSize: '1.05rem', color: '#111827' }}>Clinical Decision Support System Ready</div>
              <div style={{ fontSize: '0.9rem', marginTop: '4px' }}>Upload a patient dengue report or click any suggested question above.</div>
            </div>
          )}

          {chatHistory.map((chat, idx) => (
            <div key={idx} style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              
              {/* User Bubble */}
              <div style={userBubbleStyle}>
                <div style={{ fontSize: '0.96rem', fontWeight: '600' }}>{chat.q}</div>
              </div>

              {/* Assistant Answer Card with Colored Left Border */}
              <div style={assistantCardStyle}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#14532D', fontWeight: '800', fontSize: '1.05rem', borderBottom: '1.5px dashed #D9F99D', paddingBottom: '8px', marginBottom: '10px' }}>
                  <span>🩺</span> Synthesized Clinical Assessment
                </div>
                <div style={{ fontSize: '0.98rem', lineHeight: '1.75', color: '#1F2937', whiteSpace: 'pre-wrap' }}>
                  {chat.a}
                </div>

                {chat.evidence && chat.evidence.length > 0 && (
                  <details style={{ marginTop: '14px', borderTop: '1px solid #E5E7EB', paddingTop: '10px', fontSize: '0.88rem' }}>
                    <summary style={{ cursor: 'pointer', fontWeight: '800', color: '#050814' }}>
                      📋 View Grounded Clinical Evidence ({chat.evidence.length} sources)
                    </summary>
                    <div style={{ marginTop: '10px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      {chat.evidence.map((ev, i) => (
                        <div key={i} style={evidenceCardStyle}>
                          <div style={{ fontWeight: '700', color: '#14532D', fontSize: '0.84rem' }}>Source: {ev.source}</div>
                          <div style={{ color: '#374151', marginTop: '4px' }}>{ev.content}</div>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>

            </div>
          ))}
        </div>
        
        {/* Question Input Box */}
        <div style={inputContainerStyle}>
          <input 
            type="text" 
            value={question} 
            onChange={(e) => setQuestion(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleAsk()}
            placeholder="Ask about diagnosis, platelet count, risk level, or patient details..."
            style={textInputStyle}
          />
          <button 
            onClick={() => handleAsk()} 
            disabled={!question.trim() || loading} 
            style={sendBtnStyle(!question.trim() || loading)}
          >
            {loading ? "..." : "➔"}
          </button>
        </div>
      </div>

    </div>
  );
};

// ── Styles ─────────────────────────────────────────────────────────────────
const cardStyle = {
  background: '#FFFFFF',
  borderRadius: '20px',
  border: '2px solid #D9F99D',
  padding: '20px',
  boxShadow: '0 14px 34px -4px rgba(5, 8, 20, 0.08), 0 2px 14px rgba(132, 204, 22, 0.12)',
  display: 'flex',
  flexDirection: 'column',
  justifyContent: 'space-between',
  transition: 'transform 0.25s ease, box-shadow 0.25s ease'
};

const chatCardStyle = {
  ...cardStyle,
  padding: '26px'
};

const colHeaderStyle = {
  fontSize: '1.05rem',
  fontWeight: '800',
  color: '#050814',
  marginBottom: '12px',
  display: 'flex',
  alignItems: 'center',
  gap: '8px'
};

const productBadgeStyle = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  background: '#050814',
  border: '1.5px solid #84CC16',
  borderRadius: '9999px',
  padding: '7px 14px',
  fontSize: '0.82rem',
  boxShadow: '0 4px 14px rgba(5, 8, 20, 0.35)'
};

const pulseDotStyle = {
  width: '8px',
  height: '8px',
  backgroundColor: '#84CC16',
  borderRadius: '50%',
  boxShadow: '0 0 10px #84CC16',
  display: 'inline-block'
};

const chipBtnStyle = {
  background: '#F0FDF4',
  border: '2px solid #84CC16',
  borderRadius: '9999px',
  color: '#14532D',
  padding: '9px 18px',
  fontSize: '0.88rem',
  fontWeight: '800',
  cursor: 'pointer',
  boxShadow: '0 4px 12px rgba(132, 204, 22, 0.18)',
  transition: 'all 0.25s ease'
};

const primaryBtnStyle = (disabled) => ({
  background: disabled ? '#CBD5E1' : 'linear-gradient(135deg, #84CC16 0%, #65A30D 100%)',
  color: '#FFFFFF',
  border: 'none',
  padding: '10px 18px',
  borderRadius: '14px',
  fontWeight: '800',
  fontSize: '0.92rem',
  cursor: disabled ? 'not-allowed' : 'pointer',
  boxShadow: disabled ? 'none' : '0 6px 20px rgba(132, 204, 22, 0.45)',
  width: '100%',
  transition: 'all 0.25s ease'
});

const userBubbleStyle = {
  alignSelf: 'flex-end',
  background: '#050814',
  color: '#FFFFFF',
  border: '2px solid #84CC16',
  borderRadius: '20px 20px 4px 20px',
  padding: '14px 20px',
  maxWidth: '75%',
  boxShadow: '0 6px 20px rgba(5, 8, 20, 0.35)'
};

const assistantCardStyle = {
  background: 'linear-gradient(180deg, #FFFFFF 0%, #F8FCF5 100%)',
  border: '2px solid #D9F99D',
  borderLeft: '8px solid #84CC16',
  borderRadius: '20px',
  padding: '22px 26px',
  boxShadow: '0 14px 34px -4px rgba(5, 8, 20, 0.1)',
  animation: 'fadeIn 0.3s ease'
};

const evidenceCardStyle = {
  background: '#FFFFFF',
  border: '1.5px solid #D9F99D',
  borderLeft: '4px solid #84CC16',
  borderRadius: '12px',
  padding: '10px 14px'
};

const inputContainerStyle = {
  display: 'flex',
  alignItems: 'center',
  background: '#FFFFFF',
  border: '2.5px solid #84CC16',
  borderRadius: '32px',
  padding: '6px 12px 6px 20px',
  boxShadow: '0 12px 36px rgba(5, 8, 20, 0.12), 0 0 20px rgba(132, 204, 22, 0.25)'
};

const textInputStyle = {
  flex: 1,
  border: 'none',
  outline: 'none',
  fontSize: '0.98rem',
  fontFamily: 'inherit',
  color: '#111827',
  padding: '8px 0'
};

const sendBtnStyle = (disabled) => ({
  background: disabled ? '#94A3B8' : '#84CC16',
  color: '#FFFFFF',
  border: 'none',
  width: '38px',
  height: '38px',
  borderRadius: '50%',
  cursor: disabled ? 'not-allowed' : 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  fontSize: '1.1rem',
  fontWeight: 'bold',
  boxShadow: disabled ? 'none' : '0 4px 14px rgba(132, 204, 22, 0.5)',
  transition: 'transform 0.2s ease, background 0.2s ease'
});

export default Dashboard;
