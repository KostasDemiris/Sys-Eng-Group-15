import { Container, Typography, Grid, Box } from '@mui/material';
import { motion } from 'framer-motion';
import { 
  Description, Code, LibraryBooks, 
  ChevronRight, School, Build 
} from '@mui/icons-material';
import { 
  GlassPaper, GradientTypography, IconBox, 
  AnimatedButton, PageContainer, HeroSection 
} from '../components/StyledComponents';

function Documentation() {
  const mainSections = [
    {
      title: 'Getting Started',
      description: 'Quick start guide and basic concepts',
      icon: Description,
      items: ['Installation', 'Basic Usage', 'Configuration']
    },
    {
      title: 'API Reference',
      description: 'Detailed API documentation',
      icon: Code,
      items: ['Endpoints', 'Authentication', 'Response Formats']
    },
    {
      title: 'Tutorials',
      description: 'Step-by-step guides',
      icon: School,
      items: ['Basic Analysis', 'Advanced Features', 'Best Practices']
    }
  ];

  const quickLinks = [
    { title: 'Installation Guide', icon: Build },
    { title: 'API Documentation', icon: Code },
    { title: 'Examples', icon: LibraryBooks },
  ];

  const fadeInUp = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.5 }
  };

  return (
    <Container maxWidth="lg">
      <PageContainer>
        <motion.div {...fadeInUp}>
          <HeroSection>
            <Box sx={{ textAlign: 'center', mb: 6 }}>
              <Typography 
                variant="h3"
                gutterBottom
                sx={{
                  fontWeight: 700,
                  fontSize: '2.5rem',
                  background: 'linear-gradient(45deg, #2D3436 30%, #636E72 90%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                Documentation
              </Typography>
              <Typography 
                variant="h6"
                color="text.secondary"
                sx={{ 
                  maxWidth: '700px',
                  mx: 'auto',
                  mb: 4,
                  fontWeight: 500,
                  lineHeight: 1.6,
                  fontSize: '1.125rem',
                }}
              >
                Everything you need to know about using our satellite computer vision platform
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
              {quickLinks.map((link, index) => (
                <AnimatedButton
                  key={index}
                  variant="outlined"
                  startIcon={<link.icon />}
                  sx={{ 
                    borderRadius: '20px',
                    px: 4,
                    py: 1.5,
                  }}
                >
                  {link.title}
                </AnimatedButton>
              ))}
            </Box>
          </HeroSection>

          <Grid container spacing={4}>
            {mainSections.map((section, index) => (
              <Grid item xs={12} md={4} key={index}>
                <motion.div
                  {...fadeInUp}
                  transition={{ delay: index * 0.1 }}
                >
                  <GlassPaper sx={{ 
                    '&:hover': {
                      transform: 'translateY(-4px)',
                    }
                  }}>
                    <IconBox>
                      <section.icon sx={{ fontSize: 30 }} />
                    </IconBox>
                    
                    <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
                      {section.title}
                    </Typography>
                    <Typography color="text.secondary" sx={{ mb: 4 }}>
                      {section.description}
                    </Typography>
                    
                    {section.items.map((item, idx) => (
                      <AnimatedButton
                        key={idx}
                        fullWidth
                        sx={{
                          justifyContent: 'space-between',
                          textAlign: 'left',
                          mb: 1,
                          py: 1.5,
                          color: 'text.primary',
                          '&:hover': {
                            backgroundColor: 'rgba(9, 132, 227, 0.08)',
                          }
                        }}
                        endIcon={<ChevronRight />}
                      >
                        {item}
                      </AnimatedButton>
                    ))}
                  </GlassPaper>
                </motion.div>
              </Grid>
            ))}
          </Grid>

          <GlassPaper sx={{ 
            mt: 8, 
            p: 8, 
            textAlign: 'center',
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.85) 100%)',
          }}>
            <Typography variant="h4" gutterBottom sx={{ fontWeight: 600 }}>
              Need Help?
            </Typography>
            <Typography 
              color="text.secondary" 
              sx={{ 
                mb: 4, 
                maxWidth: '600px', 
                mx: 'auto',
                fontSize: '1.1rem',
                lineHeight: 1.8
              }}
            >
              Can't find what you're looking for? Our support team is here to help.
            </Typography>
            <AnimatedButton 
              variant="contained" 
              size="large"
              sx={{ 
                borderRadius: '30px',
                px: 6,
                py: 2,
                background: 'linear-gradient(45deg, #0984E3 30%, #74B9FF 90%)',
                boxShadow: '0 8px 24px rgba(9, 132, 227, 0.25)',
              }}
            >
              Contact Support
            </AnimatedButton>
          </GlassPaper>
        </motion.div>
      </PageContainer>
    </Container>
  );
}

export default Documentation; 