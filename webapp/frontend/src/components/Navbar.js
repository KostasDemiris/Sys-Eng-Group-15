import { AppBar, Toolbar, Typography, Box, Container } from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { styled } from '@mui/material/styles';

const StyledAppBar = styled(AppBar)(({ theme }) => ({
  backgroundColor: 'rgba(255, 255, 255, 0.9)',
  backdropFilter: 'blur(10px)',
  borderBottom: '1px solid rgba(0, 0, 0, 0.06)',
  boxShadow: 'none',
}));

const NavLink = styled(Typography)(({ theme, active }) => ({
  position: 'relative',
  padding: '8px 16px',
  cursor: 'pointer',
  color: active ? theme.palette.primary.main : theme.palette.text.secondary,
  fontWeight: active ? 600 : 500,
  fontSize: '0.95rem',
  transition: 'all 0.2s ease-in-out',
  '&::after': {
    content: '""',
    position: 'absolute',
    bottom: '-1px',
    left: active ? '10%' : '50%',
    width: active ? '80%' : '0%',
    height: '2px',
    backgroundColor: theme.palette.primary.main,
    transition: 'all 0.3s ease-in-out',
    borderRadius: '4px',
  },
  '&:hover': {
    color: theme.palette.primary.main,
    '&::after': {
      left: '10%',
      width: '80%',
    },
  },
}));

const LogoContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  cursor: 'pointer',
  transition: 'transform 0.2s ease-in-out',
  '&:hover': {
    transform: 'translateY(-1px)',
  },
}));

const LogoIcon = styled('div')(({ theme }) => ({
  width: 32,
  height: 32,
  borderRadius: '8px',
  background: 'linear-gradient(135deg, #0984E3 0%, #74B9FF 100%)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: 'white',
  fontSize: '1.2rem',
  fontWeight: 700,
}));

function Navbar() {
  const navigate = useNavigate();
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Documentation' },
    { path: '/processing', label: 'Online Processing' },
    { path: '/about', label: 'About' },
  ];

  return (
    <StyledAppBar position="sticky">
      <Container maxWidth="lg">
        <Toolbar disableGutters sx={{ height: 72 }}>
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
          >
            <LogoContainer onClick={() => navigate('/')} sx={{ mr: 4 }}>
              <img src="/favicon.ico" alt="SightLink Logo" style={{ width: 32, height: 32 }} />
              <Typography 
                variant="h6" 
                component="div" 
                sx={{ 
                  fontWeight: 700,
                  background: 'linear-gradient(45deg, #2D3436 30%, #636E72 90%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                SightLink
              </Typography>
            </LogoContainer>
          </motion.div>

          <Box sx={{ flexGrow: 1 }} />

          <Box 
            component={motion.div}
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            sx={{ 
              display: 'flex', 
              gap: 1,
              alignItems: 'center',
              height: '100%'
            }}
          >
            {navItems.map((item, index) => (
              <NavLink
                key={index}
                active={location.pathname === item.path ? 1 : 0}
                onClick={() => navigate(item.path)}
                sx={{
                  opacity: location.pathname === item.path ? 1 : 0.8,
                }}
              >
                {item.label}
              </NavLink>
            ))}
          </Box>
        </Toolbar>
      </Container>
    </StyledAppBar>
  );
}

export default Navbar; 