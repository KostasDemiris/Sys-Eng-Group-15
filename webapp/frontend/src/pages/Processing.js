import { useState } from 'react';
import React from 'react';
import { 
  Container, Typography, Box, Button, ToggleButton, 
  ToggleButtonGroup, Paper, IconButton, LinearProgress, 
  Grid, Fade, Stepper, Step, StepLabel, StepContent,
  useTheme
} from '@mui/material';
import { 
  CloudUpload as UploadIcon,
  Close as CloseIcon,
  CheckCircle as CheckCircleIcon,
  Article as ArticleIcon,
  NavigateNext as NextIcon,
  Map as MapIcon,
  Category as CategoryIcon,
  Upload as UploadFileIcon,
  Chair as ChairIcon,
  TableRows as TableRowsIcon,
  Escalator as EscalatorIcon,
  DirectionsWalk as DirectionsWalkIcon,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { styled } from '@mui/material/styles';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  height: '100%',
  backgroundColor: 'rgba(255, 255, 255, 0.95)',
  backdropFilter: 'blur(20px)',
  borderRadius: theme.shape.borderRadius * 2,
  transition: 'all 0.3s ease-in-out',
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.08)',
  border: '1px solid rgba(255, 255, 255, 0.4)',
}));

const StyledStepIcon = styled('div')(({ theme, completed, active }) => ({
  width: 48,
  height: 48,
  borderRadius: '14px',
  backgroundColor: completed 
    ? 'rgba(9, 132, 227, 0.9)'
    : active 
    ? 'rgba(9, 132, 227, 0.1)'
    : theme.palette.grey[50],
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: completed 
    ? 'white'
    : active 
    ? theme.palette.primary.main
    : theme.palette.grey[400],
  transition: 'all 0.3s ease',
  boxShadow: completed || active 
    ? '0 4px 12px rgba(9, 132, 227, 0.2)'
    : 'none',
  transform: active ? 'scale(1.05)' : 'scale(1)',
}));

const StyledToggleButton = styled(ToggleButton)(({ theme }) => ({
  borderRadius: '16px !important',
  padding: '24px',
  minHeight: '160px',
  border: '1px solid rgba(0, 0, 0, 0.08)',
  transition: 'all 0.3s ease',
  backgroundColor: 'rgba(255, 255, 255, 0.8)',
  flex: 1,
  boxShadow: '0 2px 12px rgba(0, 0, 0, 0.05)',
  '&.Mui-selected': {
    backgroundColor: 'rgba(9, 132, 227, 0.04)',
    border: '2px solid #0984E3',
    boxShadow: '0 8px 24px rgba(9, 132, 227, 0.15)',
  },
  '&:hover': {
    backgroundColor: 'rgba(9, 132, 227, 0.04)',
    transform: 'translateY(-4px)',
    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.12)',
    borderColor: theme.palette.primary.main,
  }
}));

const AnimatedButton = styled(Button)(({ theme }) => ({
  borderRadius: '12px',
  padding: '12px 24px',
  transition: 'all 0.3s ease',
  position: 'relative',
  overflow: 'hidden',
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

const UploadArea = styled(Box)(({ theme, isDragActive }) => ({
  padding: theme.spacing(6),
  borderRadius: theme.shape.borderRadius * 2,
  border: `2px dashed ${isDragActive ? theme.palette.primary.main : 'rgba(0, 0, 0, 0.1)'}`,
  backgroundColor: isDragActive ? 'rgba(9, 132, 227, 0.04)' : 'rgba(255, 255, 255, 0.6)',
  transition: 'all 0.3s ease',
  cursor: 'pointer',
  textAlign: 'center',
  '&:hover': {
    backgroundColor: 'rgba(9, 132, 227, 0.02)',
    borderColor: theme.palette.primary.main,
    transform: 'translateY(-2px)',
  }
}));

function CustomStepIcon({ icon, completed, active }) {
  const getStepIcon = (step) => {
    switch (step) {
      case 1:
        return <MapIcon fontSize="small" />;
      case 2:
        return <CategoryIcon fontSize="small" />;
      case 3:
        return <UploadFileIcon fontSize="small" />;
      default:
        return null;
    }
  };

  return (
    <StyledStepIcon completed={completed} active={active}>
      {completed ? <CheckCircleIcon fontSize="small" /> : getStepIcon(icon)}
    </StyledStepIcon>
  );
}

const IconContainer = styled(Box)(({ selected }) => ({
  width: 64,
  height: 64,
  borderRadius: '16px',
  background: selected 
    ? 'linear-gradient(135deg, #0984E3 0%, #74B9FF 100%)'
    : 'rgba(0, 0, 0, 0.04)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  marginBottom: 12,
  marginLeft: 'auto',
  marginRight: 'auto',
  transition: 'all 0.3s ease',
  boxShadow: selected
    ? '0 8px 20px rgba(9, 132, 227, 0.2)'
    : 'none',
  transform: selected ? 'scale(1.05)' : 'scale(1)',
  '&:hover': {
    transform: 'scale(1.05)',
  }
}));

const ButtonContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'flex-end',
  gap: theme.spacing(2),
  marginTop: 'auto',
  paddingTop: theme.spacing(3),
}));

const ObjectToggleButton = styled(ToggleButton)(({ theme }) => ({
  borderRadius: '8px !important',
  padding: '8px 16px',
  border: '1px solid rgba(0, 0, 0, 0.08)',
  transition: 'all 0.3s ease',
  backgroundColor: 'rgba(255, 255, 255, 0.8)',
  flex: 1,
  '&.Mui-selected': {
    backgroundColor: 'rgba(9, 132, 227, 0.08)',
    color: theme.palette.primary.main,
    fontWeight: 600,
    border: '1px solid rgba(9, 132, 227, 0.3)',
    boxShadow: '0 4px 12px rgba(9, 132, 227, 0.12)',
    '&::before': {
      content: '"✓"',
      marginRight: theme.spacing(1),
      fontWeight: 'bold',
    }
  },
  '&:hover': {
    backgroundColor: 'rgba(9, 132, 227, 0.04)',
    transform: 'translateY(-2px)',
  }
}));

const ContinueButton = styled(Button)(({ theme }) => ({
  flex: 1,
  padding: '12px 24px',
  borderRadius: '10px',
  background: 'linear-gradient(45deg, #0984E3 30%, #74B9FF 90%)',
  boxShadow: '0 8px 24px rgba(9, 132, 227, 0.25)',
  transition: 'all 0.3s ease',
  '& .arrow-icon': {
    opacity: 0,
    width: 0,
    marginLeft: 0,
    transition: 'all 0.3s ease',
  },
  '&:hover': {
    '& .arrow-icon': {
      opacity: 1,
      width: '24px',
      marginLeft: theme.spacing(1),
    }
  }
}));

const DetectionObjectCard = styled(Paper)(({ theme, selected }) => ({
  padding: theme.spacing(2.5),
  borderRadius: '16px',
  border: '1px solid rgba(0, 0, 0, 0.08)',
  backgroundColor: selected ? 'rgba(9, 132, 227, 0.04)' : 'rgba(255, 255, 255, 0.8)',
  cursor: 'pointer',
  transition: 'all 0.3s ease',
  display: 'flex',
  alignItems: 'center',
  position: 'relative',
  overflow: 'hidden',
  boxShadow: selected 
    ? '0 8px 24px rgba(9, 132, 227, 0.15)'
    : '0 2px 12px rgba(0, 0, 0, 0.05)',
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.12)',
    borderColor: selected ? '#0984E3' : 'rgba(9, 132, 227, 0.3)',
  },
  ...(selected && {
    border: '2px solid #0984E3',
    backgroundColor: 'rgba(9, 132, 227, 0.04)',
  })
}));

const ObjectIcon = styled(Box)(({ theme, selected }) => ({
  width: 48,
  height: 48,
  borderRadius: '12px',
  background: selected 
    ? 'linear-gradient(135deg, #0984E3 0%, #74B9FF 100%)'
    : 'rgba(0, 0, 0, 0.04)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  marginRight: theme.spacing(2),
  transition: 'all 0.3s ease',
  flexShrink: 0,
  '& svg': {
    fontSize: 24,
    color: selected ? 'white' : '#666666',
    transition: 'all 0.3s ease',
  }
}));

const FileCard = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  marginBottom: theme.spacing(1.5),
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  borderRadius: '20px',
  border: '1px solid rgba(0, 0, 0, 0.08)',
  backgroundColor: 'rgba(255, 255, 255, 0.8)',
  transition: 'all 0.3s ease',
  transform: 'translateX(0)',
  '&:hover': {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    transform: 'translateX(8px)',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.05)',
  }
}));

const ResultButton = styled(Button)(({ theme }) => ({
  padding: '12px 32px',
  borderRadius: '30px',
  transition: 'all 0.3s ease',
  fontSize: '1rem',
  fontWeight: 600,
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: '0 8px 24px rgba(0, 0, 0, 0.1)',
  }
}));

function Processing() {
  const theme = useTheme();
  const [activeStep, setActiveStep] = useState(0);
  const [selectedObjects, setSelectedObjects] = useState(['Benches']);
  const [inputType, setInputType] = useState('DigiMap');
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processSuccess, setProcessSuccess] = useState(false);
  const [progress, setProgress] = useState(0);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (acceptedFiles) => {
      setUploadedFiles([...uploadedFiles, ...acceptedFiles]);
    }
  });

  const handleStartProcessing = () => {
    setIsProcessing(true);
    const timer = setInterval(() => {
      setProgress((oldProgress) => {
        const newProgress = oldProgress + 10;
        if (newProgress === 100) {
          clearInterval(timer);
          setTimeout(() => {
            setIsProcessing(false);
            setProcessSuccess(true);
          }, 500);
        }
        return newProgress;
      });
    }, 500);
  };

  const steps = [
    {
      label: 'Select Input Type',
      description: 'Choose the type of data you want to process',
      content: (
        <Box sx={{ py: 4 }}>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 700 }}>
            Choose your data source type
          </Typography>
          <Typography color="text.secondary" sx={{ mb: 4, fontSize: '1rem' }}>
            Select the appropriate data format for your analysis
          </Typography>
          <ToggleButtonGroup
            value={inputType}
            exclusive
            onChange={(e, value) => value && setInputType(value)}
            fullWidth
            sx={{ mb: 4, gap: 2 }}
          >
            <StyledToggleButton value="DigiMap">
              <Box sx={{ textAlign: 'center' }}>
                <IconContainer selected={inputType === 'DigiMap'}>
                  <MapIcon sx={{ 
                    fontSize: 32, 
                    color: inputType === 'DigiMap' ? 'white' : '#666666',
                    transition: 'all 0.3s ease',
                  }} />
                </IconContainer>
                <Typography 
                  variant="subtitle1" 
                  sx={{ 
                    fontWeight: 600, 
                    mb: 0.5,
                    color: inputType === 'DigiMap' ? '#0984E3' : '#000000',
                  }}
                >
                  DigiMap Data
                </Typography>
                <Typography 
                  variant="body2" 
                  sx={{ 
                    color: inputType === 'DigiMap' ? '#0984E3' : 'text.secondary',
                    opacity: 0.8,
                    fontSize: '0.875rem'
                  }}
                >
                  Process DigiMap format files
                </Typography>
              </Box>
            </StyledToggleButton>
            <StyledToggleButton value="Custom">
              <Box sx={{ textAlign: 'center' }}>
                <IconContainer selected={inputType === 'Custom'}>
                  <CategoryIcon sx={{ 
                    fontSize: 40, 
                    color: inputType === 'Custom' ? 'white' : '#666666',
                    transition: 'all 0.3s ease',
                  }} />
                </IconContainer>
                <Typography 
                  variant="h6" 
                  sx={{ 
                    fontWeight: 600, 
                    mb: 1,
                    color: inputType === 'Custom' ? '#0984E3' : '#000000',
                  }}
                >
                  Custom Data
                </Typography>
                <Typography 
                  variant="body2" 
                  sx={{ 
                    color: inputType === 'Custom' ? '#0984E3' : 'text.secondary',
                    opacity: 0.8
                  }}
                >
                  Process custom format files
                </Typography>
              </Box>
            </StyledToggleButton>
          </ToggleButtonGroup>
          <ButtonContainer>
            <ContinueButton
              variant="contained"
              onClick={() => setActiveStep(1)}
              disabled={!inputType}
              sx={{ 
                flex: '0 1 50%',
              }}
            >
              Continue
              <NextIcon className="arrow-icon" />
            </ContinueButton>
          </ButtonContainer>
        </Box>
      )
    },
    {
      label: 'Choose Detection Objects',
      description: 'Select objects to detect in your data',
      content: (
        <Box sx={{ py: 4 }}>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 700 }}>
            Select Detection Objects
          </Typography>
          <Typography color="text.secondary" sx={{ mb: 4, fontSize: '1rem' }}>
            Choose the objects you want to detect in your data
          </Typography>
          
          <Grid container spacing={2} sx={{ mb: 4 }}>
            {[
              { id: 'Benches', icon: <ChairIcon />, label: 'Benches', description: 'Detect public seating areas' },
              { id: 'Zebra Cross', icon: <TableRowsIcon />, label: 'Zebra Crossings', description: 'Identify pedestrian crossings' },
              { id: 'Ramps', icon: <EscalatorIcon />, label: 'Ramps', description: 'Locate accessibility ramps' },
              { id: 'Pavements', icon: <DirectionsWalkIcon />, label: 'Pavements', description: 'Detect walking paths' }
            ].map((item) => (
              <Grid item xs={12} sm={6} key={item.id}>
                <DetectionObjectCard
                  selected={selectedObjects.includes(item.id)}
                  onClick={() => {
                    setSelectedObjects(prev => 
                      prev.includes(item.id)
                        ? prev.filter(i => i !== item.id)
                        : [...prev, item.id]
                    );
                  }}
                  elevation={selectedObjects.includes(item.id) ? 2 : 1}
                >
                  <ObjectIcon selected={selectedObjects.includes(item.id)}>
                    {React.cloneElement(item.icon, { 
                      sx: { 
                        fontSize: 24,
                        color: selectedObjects.includes(item.id) ? 'white' : '#666666'
                      } 
                    })}
                  </ObjectIcon>
                  <Box sx={{ flex: 1 }}>
                    <Typography 
                      variant="subtitle1" 
                      sx={{ 
                        fontWeight: 600,
                        color: selectedObjects.includes(item.id) ? '#0984E3' : 'text.primary'
                      }}
                    >
                      {item.label}
                    </Typography>
                    <Typography 
                      variant="body2" 
                      color="text.secondary"
                      sx={{ opacity: 0.8 }}
                    >
                      {item.description}
                    </Typography>
                  </Box>
                </DetectionObjectCard>
              </Grid>
            ))}
          </Grid>

          <ButtonContainer>
            <Button
              variant="outlined"
              onClick={() => setActiveStep(0)}
              sx={{
                flex: 1,
                py: 1.5,
                borderRadius: '10px',
              }}
            >
              Back
            </Button>
            <ContinueButton
              variant="contained"
              onClick={() => setActiveStep(2)}
              disabled={selectedObjects.length === 0}
            >
              Continue
              <NextIcon className="arrow-icon" />
            </ContinueButton>
          </ButtonContainer>
        </Box>
      )
    },
    {
      label: 'Upload Files',
      description: 'Upload your files for processing',
      content: (
        <Box sx={{ py: 4 }}>
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 700 }}>
            Upload Your Files
          </Typography>
          <Typography color="text.secondary" sx={{ mb: 4, fontSize: '1rem' }}>
            Drag and drop your files or click to browse
          </Typography>

          <StyledPaper
            {...getRootProps()}
            sx={{
              cursor: 'pointer',
              textAlign: 'center',
              py: uploadedFiles.length > 0 ? 4 : 8,
              px: 4,
              background: isDragActive 
                ? 'rgba(9, 132, 227, 0.04)' 
                : uploadedFiles.length > 0
                ? 'rgba(255, 255, 255, 0.7)'
                : 'rgba(255, 255, 255, 0.9)',
              border: `2px dashed ${
                isDragActive 
                  ? '#0984E3' 
                  : uploadedFiles.length > 0
                  ? 'rgba(9, 132, 227, 0.1)'
                  : 'rgba(9, 132, 227, 0.2)'
              }`,
              transition: 'all 0.3s ease',
              '&:hover': {
                border: '2px dashed #0984E3',
                backgroundColor: 'rgba(9, 132, 227, 0.04)',
                transform: 'translateY(-2px)',
              }
            }}
          >
            <input {...getInputProps()} />
            {uploadedFiles.length > 0 ? (
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 2 }}>
                <Box
                  sx={{
                    width: 48,
                    height: 48,
                    borderRadius: '12px',
                    background: 'linear-gradient(135deg, rgba(9, 132, 227, 0.1) 0%, rgba(116, 185, 255, 0.1) 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <UploadIcon sx={{ 
                    fontSize: 24, 
                    color: 'rgba(9, 132, 227, 0.6)',
                  }} />
                </Box>
                <Box sx={{ textAlign: 'left' }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                    Add more files
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    or drop them here
                  </Typography>
                </Box>
              </Box>
            ) : (
              <>
                <Box
                  sx={{
                    width: 80,
                    height: 80,
                    borderRadius: '20px',
                    background: 'linear-gradient(135deg, rgba(9, 132, 227, 0.1) 0%, rgba(116, 185, 255, 0.1) 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    margin: '0 auto',
                    mb: 3,
                  }}
                >
                  <UploadIcon sx={{ 
                    fontSize: 40, 
                    color: isDragActive ? '#0984E3' : 'rgba(9, 132, 227, 0.6)',
                    transition: 'all 0.3s ease',
                  }} />
                </Box>
                <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                  {isDragActive ? 'Drop files here' : 'Upload Files'}
                </Typography>
                <Typography color="text.secondary" sx={{ mb: 2 }}>
                  {isDragActive ? 'Release to upload files' : 'Drag and drop files here or click to browse'}
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                  Supported formats: .jpg, .png, .tiff
                </Typography>
              </>
            )}
          </StyledPaper>

          {uploadedFiles.length > 0 && (
            <Box sx={{ mt: 4 }}>
              <Box sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'space-between',
                mb: 2 
              }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                  Uploaded Files
                </Typography>
                <Typography 
                  variant="body2" 
                  color="text.secondary"
                  sx={{ 
                    backgroundColor: 'rgba(9, 132, 227, 0.08)',
                    px: 1.5,
                    py: 0.5,
                    borderRadius: '20px',
                    fontWeight: 500
                  }}
                >
                  {uploadedFiles.length} {uploadedFiles.length === 1 ? 'file' : 'files'}
                </Typography>
              </Box>
              {uploadedFiles.map((file, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ 
                    duration: 0.3,
                    delay: index * 0.1,
                    ease: "easeOut"
                  }}
                >
                  <FileCard>
                    <Box sx={{ display: 'flex', alignItems: 'center', flex: 1 }}>
                      <Box
                        sx={{
                          width: 40,
                          height: 40,
                          borderRadius: '14px',
                          backgroundColor: 'rgba(9, 132, 227, 0.08)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          mr: 2,
                          transition: 'all 0.3s ease',
                        }}
                      >
                        <ArticleIcon 
                          sx={{ 
                            color: '#0984E3',
                            fontSize: 20,
                          }} 
                        />
                      </Box>
                      <Box sx={{ flex: 1, minWidth: 0 }}>
                        <Typography 
                          variant="subtitle2" 
                          sx={{ 
                            fontWeight: 600,
                            color: 'text.primary',
                            mb: 0.5,
                            textAlign: 'left',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap'
                          }}
                        >
                          {file.name}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                          <Typography 
                            variant="caption" 
                            sx={{ 
                              color: 'text.secondary',
                              backgroundColor: 'rgba(0, 0, 0, 0.04)',
                              px: 1.5,
                              py: 0.5,
                              borderRadius: '20px',
                              fontWeight: 500,
                              textTransform: 'uppercase',
                              letterSpacing: '0.02em'
                            }}
                          >
                            {file.name.split('.').pop().toUpperCase()}
                          </Typography>
                          <Typography 
                            variant="caption" 
                            color="text.secondary"
                            sx={{ flexShrink: 0 }}
                          >
                            {(file.size / 1024 / 1024).toFixed(2)} MB
                          </Typography>
                        </Box>
                      </Box>
                    </Box>
                    <motion.div
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                    >
                      <IconButton 
                        onClick={(e) => {
                          e.stopPropagation();
                          const element = e.currentTarget.parentElement.parentElement.parentElement;
                          element.style.transform = 'translateX(-100%)';
                          element.style.opacity = '0';
                          setTimeout(() => {
                            setUploadedFiles(files => files.filter((_, i) => i !== index));
                          }, 300);
                        }}
                        sx={{
                          width: 32,
                          height: 32,
                          color: 'text.secondary',
                          transition: 'all 0.2s ease',
                          '&:hover': {
                            backgroundColor: 'error.light',
                            color: 'error.main',
                          }
                        }}
                      >
                        <CloseIcon sx={{ fontSize: 18 }} />
                      </IconButton>
                    </motion.div>
                  </FileCard>
                </motion.div>
              ))}
            </Box>
          )}

          <ButtonContainer>
            <Button
              variant="outlined"
              onClick={() => setActiveStep(1)}
              sx={{
                flex: 1,
                py: 1.5,
                borderRadius: '10px',
              }}
            >
              Back
            </Button>
            <ContinueButton
              variant="contained"
              onClick={handleStartProcessing}
              disabled={uploadedFiles.length === 0}
              sx={{ flex: '0 1 50%' }}
            >
              Start Processing
              <NextIcon className="arrow-icon" />
            </ContinueButton>
          </ButtonContainer>
        </Box>
      )
    }
  ];

  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 8 }}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Box sx={{ textAlign: 'center', mb: 6 }}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.5 }}
            >
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
                Online Processing
              </Typography>
              <Typography 
                variant="h6" 
                color="text.secondary"
                sx={{ 
                  maxWidth: '700px',
                  mx: 'auto',
                  mb: 3,
                  fontWeight: 500,
                  lineHeight: 1.6,
                  fontSize: '1.125rem',
                }}
              >
                Transform your satellite imagery with our advanced computer vision technology
              </Typography>
            </motion.div>
          </Box>

          {!isProcessing && !processSuccess ? (
            <Grid container spacing={4}>
              <Grid item xs={12} md={4}>
                <StyledPaper 
                  sx={{ 
                    position: 'sticky', 
                    top: 90,
                    background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.85) 100%)',
                  }}
                >
                  <Stepper 
                    activeStep={activeStep} 
                    orientation="vertical"
                    sx={{
                      '.MuiStepConnector-line': {
                        minHeight: 40,
                      }
                    }}
                  >
                    {steps.map((step, index) => (
                      <Step key={index}>
                        <StepLabel 
                          StepIconComponent={(props) => (
                            <CustomStepIcon {...props} active={activeStep === index} />
                          )}
                        >
                          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                            {step.label}
                          </Typography>
                          {activeStep === index && (
                            <motion.div
                              initial={{ opacity: 0, y: -10 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ duration: 0.3 }}
                            >
                              <Typography 
                                variant="body2" 
                                color="text.secondary"
                                sx={{ 
                                  mt: 0.5,
                                  opacity: 0.8
                                }}
                              >
                                {step.description}
                              </Typography>
                            </motion.div>
                          )}
                        </StepLabel>
                      </Step>
                    ))}
                  </Stepper>
                </StyledPaper>
              </Grid>
              <Grid item xs={12} md={8}>
                <StyledPaper>
                  {steps[activeStep].content}
                </StyledPaper>
              </Grid>
            </Grid>
          ) : isProcessing ? (
            <StyledPaper sx={{ p: 6, textAlign: 'center' }}>
              <Typography variant="h5" gutterBottom>
                Processing Files...
              </Typography>
              <Box sx={{ width: '100%', mt: 4 }}>
                <LinearProgress 
                  variant="determinate" 
                  value={progress} 
                  sx={{ 
                    height: 8, 
                    borderRadius: 4,
                    backgroundColor: 'rgba(9, 132, 227, 0.1)',
                    '& .MuiLinearProgress-bar': {
                      background: 'linear-gradient(90deg, #0984E3 0%, #74B9FF 100%)',
                    }
                  }} 
                />
                <Typography 
                  variant="body2" 
                  color="text.secondary"
                  sx={{ mt: 2 }}
                >
                  {`${progress}% Complete`}
                </Typography>
              </Box>
            </StyledPaper>
          ) : (
            <StyledPaper 
              sx={{ 
                p: 8,
                textAlign: 'center',
                background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.85) 100%)',
              }}
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.5 }}
              >
                <Box
                  sx={{
                    width: 100,
                    height: 100,
                    borderRadius: '30px',
                    background: 'linear-gradient(135deg, #4CAF50 0%, #81C784 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    margin: '0 auto',
                    mb: 4,
                    boxShadow: '0 12px 32px rgba(76, 175, 80, 0.2)',
                  }}
                >
                  <CheckCircleIcon 
                    sx={{ 
                      fontSize: 50,
                      color: 'white',
                    }} 
                  />
                </Box>
                <Typography 
                  variant="h3" 
                  gutterBottom 
                  sx={{ 
                    fontWeight: 700,
                    background: 'linear-gradient(45deg, #2D3436 30%, #636E72 90%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    mb: 2,
                  }}
                >
                  Success!
                </Typography>
                <Typography 
                  variant="h6" 
                  sx={{ 
                    color: 'text.secondary',
                    mb: 6,
                    maxWidth: '500px',
                    mx: 'auto',
                    lineHeight: 1.6,
                  }}
                >
                  Your files have been successfully processed and are ready for download
                </Typography>
                <Box 
                  sx={{ 
                    display: 'flex', 
                    justifyContent: 'center', 
                    gap: 2,
                  }}
                >
                  <Button
                    variant="outlined"
                    onClick={() => {
                      setProcessSuccess(false);
                      setUploadedFiles([]);
                      setActiveStep(0);
                    }}
                    sx={{
                      flex: 1,
                      py: 1.5,
                      borderRadius: '10px',
                      maxWidth: 280,
                    }}
                  >
                    Process New Files
                  </Button>
                  <ContinueButton
                    variant="contained"
                    sx={{ 
                      flex: '0 1 280px',
                    }}
                  >
                    Download Results
                    <NextIcon className="arrow-icon" />
                  </ContinueButton>
                </Box>
              </motion.div>
            </StyledPaper>
          )}
        </motion.div>
      </Box>
    </Container>
  );
}

export default Processing; 