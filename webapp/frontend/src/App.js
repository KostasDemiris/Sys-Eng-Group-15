import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { Box } from '@mui/material';
import CssBaseline from '@mui/material/CssBaseline';
import theme from './theme';
import Navbar from './components/Navbar';
import Documentation from './pages/Documentation';
import Processing from './pages/Processing';
import About from './pages/About';
import './App.css';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ 
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #F9FAFB 0%, #F0F2F5 100%)',
        }}>
          <Navbar />
          <Routes>
            <Route path="/" element={<Documentation />} />
            <Route path="/processing" element={<Processing />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;