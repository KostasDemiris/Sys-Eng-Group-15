import { Box, Container, Typography, Grid, IconButton, Link } from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  GitHub as GitHubIcon,
  LinkedIn as LinkedInIcon,
  Twitter as TwitterIcon,
  Email as EmailIcon
} from '@mui/icons-material';

const FooterWrapper = styled(Box)(({ theme }) => ({
  backgroundColor: 'rgba(255, 255, 255, 0.9)',
  backdropFilter: 'blur(20px)',
  borderTop: '1px solid rgba(0, 0, 0, 0.08)',
  padding: theme.spacing(6, 0),
  marginTop: 'auto',
}));

const FooterLink = styled(Link)(({ theme }) => ({
  color: theme.palette.text.secondary,
  textDecoration: 'none',
  transition: 'all 0.2s ease',
  '&:hover': {
    color: theme.palette.primary.main,
  }
}));

const SocialButton = styled(IconButton)(({ theme }) => ({
  color: theme.palette.text.secondary,
  transition: 'all 0.2s ease',
  '&:hover': {
    transform: 'translateY(-2px)',
    color: theme.palette.primary.main,
    backgroundColor: 'rgba(9, 132, 227, 0.08)',
  }
}));

function Footer() {
  const footerLinks = {
    'Product': [
      { name: 'Features', href: '#' },
      { name: 'Documentation', href: '/' },
      { name: 'Processing', href: '/processing' },
      { name: 'About', href: '/about' },
    ],
    'Resources': [
      { name: 'Getting Started', href: '#' },
      { name: 'API Reference', href: '#' },
      { name: 'Examples', href: '#' },
      { name: 'Support', href: '#' },
    ],
    'Company': [
      { name: 'Blog', href: '#' },
      { name: 'Careers', href: '#' },
      { name: 'Contact', href: '#' },
      { name: 'Privacy Policy', href: '#' },
    ],
  };

  return (
    <FooterWrapper>
      <Container maxWidth="lg">
        <Grid container spacing={4}>
          <Grid item xs={12} md={4}>
            <Box sx={{ mb: 4 }}>
              <Typography 
                variant="h6" 
                sx={{ 
                  fontWeight: 700,
                  background: 'linear-gradient(45deg, #2D3436 30%, #636E72 90%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  mb: 2,
                }}
              >
                SightLink
              </Typography>
              <Typography 
                variant="body2" 
                color="text.secondary"
                sx={{ mb: 2, maxWidth: 300 }}
              >
                Advanced satellite imagery analysis with computer vision technology
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <SocialButton size="small">
                  <GitHubIcon fontSize="small" />
                </SocialButton>
                <SocialButton size="small">
                  <LinkedInIcon fontSize="small" />
                </SocialButton>
                <SocialButton size="small">
                  <TwitterIcon fontSize="small" />
                </SocialButton>
                <SocialButton size="small">
                  <EmailIcon fontSize="small" />
                </SocialButton>
              </Box>
            </Box>
          </Grid>
          {Object.entries(footerLinks).map(([category, links]) => (
            <Grid item xs={12} sm={6} md={2} key={category}>
              <Typography 
                variant="subtitle2" 
                sx={{ 
                  fontWeight: 600,
                  mb: 2,
                }}
              >
                {category}
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                {links.map((link) => (
                  <FooterLink 
                    href={link.href} 
                    key={link.name}
                    variant="body2"
                  >
                    {link.name}
                  </FooterLink>
                ))}
              </Box>
            </Grid>
          ))}
        </Grid>
        <Box 
          sx={{ 
            mt: 6, 
            pt: 3, 
            borderTop: '1px solid rgba(0, 0, 0, 0.08)',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: 2,
          }}
        >
          <Typography variant="body2" color="text.secondary">
            © 2024 SightLink. All rights reserved.
          </Typography>
          <Box sx={{ display: 'flex', gap: 3 }}>
            <FooterLink variant="body2" href="#">
              Terms
            </FooterLink>
            <FooterLink variant="body2" href="#">
              Privacy
            </FooterLink>
            <FooterLink variant="body2" href="#">
              Cookies
            </FooterLink>
          </Box>
        </Box>
      </Container>
    </FooterWrapper>
  );
}

export default Footer; 