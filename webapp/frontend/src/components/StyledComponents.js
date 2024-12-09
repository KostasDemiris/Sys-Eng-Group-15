import { styled } from '@mui/material/styles';
import { Paper, Box, Button, Typography, Grid } from '@mui/material';

export const GlassPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  height: '100%',
  backgroundColor: 'rgba(255, 255, 255, 0.95)',
  backdropFilter: 'blur(20px)',
  borderRadius: theme.shape.borderRadius * 2,
  transition: 'all 0.3s ease-in-out',
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.08)',
  border: '1px solid rgba(255, 255, 255, 0.4)',
  '&:hover': {
    boxShadow: '0 12px 40px rgba(0, 0, 0, 0.12)',
  }
}));

export const GradientTypography = styled('div')(({ theme }) => ({
  fontWeight: 800,
  background: 'linear-gradient(45deg, #2D3436 30%, #636E72 90%)',
  WebkitBackgroundClip: 'text',
  WebkitTextFillColor: 'transparent',
  letterSpacing: '-0.02em',
}));

export const AnimatedButton = styled(Button)(({ theme }) => ({
  borderRadius: '12px',
  padding: '12px 24px',
  transition: 'all 0.3s ease',
  position: 'relative',
  overflow: 'hidden',
  textTransform: 'none',
  fontWeight: 600,
  '&:before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
    background: 'linear-gradient(45deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%)',
    transition: 'all 0.3s ease',
  },
  '&:hover': {
    transform: 'translateY(-2px)',
    '&:before': {
      transform: 'translateX(100%)',
    }
  }
}));

export const IconBox = styled(Box)(({ theme }) => ({
  width: 60,
  height: 60,
  borderRadius: '16px',
  background: 'linear-gradient(135deg, #0984E3 0%, #74B9FF 100%)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: 'white',
  marginBottom: theme.spacing(3),
  boxShadow: '0 8px 20px rgba(9, 132, 227, 0.2)',
}));

export const PageContainer = styled(Box)(({ theme }) => ({
  paddingTop: theme.spacing(8),
  paddingBottom: theme.spacing(8),
}));

export const HeroSection = styled(Box)(({ theme }) => ({
  textAlign: 'center',
  marginBottom: theme.spacing(10),
}));

export const PageTitle = styled(Typography)(({ theme }) => ({
  fontSize: '2.5rem',
  fontWeight: 800,
  marginBottom: theme.spacing(2),
  background: 'linear-gradient(45deg, #2D3436 30%, #636E72 90%)',
  WebkitBackgroundClip: 'text',
  WebkitTextFillColor: 'transparent',
  textAlign: 'center',
  letterSpacing: '-0.02em',
}));

export const PageSubtitle = styled(Typography)(({ theme }) => ({
  fontSize: '1.25rem',
  color: theme.palette.text.secondary,
  textAlign: 'center',
  maxWidth: '800px',
  margin: '0 auto',
  marginBottom: theme.spacing(6),
  lineHeight: 1.6,
}));

export const ContentSection = styled(Box)(({ theme }) => ({
  marginBottom: theme.spacing(8),
}));

export const CardGrid = styled(Grid)(({ theme }) => ({
  '& .MuiGrid-item': {
    display: 'flex',
  },
})); 