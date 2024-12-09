import { Box, Container } from '@mui/material';
import { motion } from 'framer-motion';
import { styled } from '@mui/material/styles';

const PageWrapper = styled(Box)(({ theme }) => ({
  minHeight: '100vh',
  background: 'linear-gradient(135deg, #F9FAFB 0%, #F0F2F5 100%)',
  paddingTop: theme.spacing(12),
  paddingBottom: theme.spacing(12),
}));

const ContentWrapper = styled(Container)(({ theme }) => ({
  position: 'relative',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    width: '100vw',
    height: '100vh',
    background: 'radial-gradient(circle at center, rgba(9, 132, 227, 0.03) 0%, transparent 70%)',
    pointerEvents: 'none',
    zIndex: -1,
  }
}));

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5 }
};

function PageLayout({ children }) {
  return (
    <PageWrapper>
      <ContentWrapper maxWidth="lg">
        <motion.div {...fadeInUp}>
          {children}
        </motion.div>
      </ContentWrapper>
    </PageWrapper>
  );
}

export default PageLayout; 