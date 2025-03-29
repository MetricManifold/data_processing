import React, { useState } from 'react';
import { Typography, Box, Button } from '@mui/material';
import InspectDisplay from './InspectDisplay';
import ImageDisplay from './ImageDisplay';

interface DataDisplayProps {
    directory: string;
}

const DataDisplay: React.FC<DataDisplayProps> = ({ directory }) => {
    const [inspecting, setInspecting] = useState<boolean>(false);
    const [currentIndex, setCurrentIndex] = useState<number>(0);

    const handleInspect = () => {
        setInspecting(!inspecting);
    };

    const handleNext = () => {
        setCurrentIndex((prevIndex) => (prevIndex + 1));
    };

    const handlePrev = () => {
        setCurrentIndex((prevIndex) => (prevIndex - 1));
    };

    return (
        <Box>
            <Typography variant="h5" gutterBottom>
                Images for {directory}
            </Typography>
            <Box display="flex" justifyContent="space-between" m={2}>
                <Button variant="contained" color="primary" onClick={handlePrev}>
                    Previous
                </Button>
                <Button variant="contained" color="primary" onClick={handleInspect}>
                    {inspecting ? <> Field</> : <>Inspect</>}
                </Button>
                <Button variant="contained" color="primary" onClick={handleNext}>
                    Next
                </Button>
            </Box>

            <Box m={2}>
                {inspecting ? (<InspectDisplay directory={directory} currentIndex={currentIndex} />) : (<ImageDisplay directory={directory} currentIndex={currentIndex} />)}
            </Box>

        </Box >

    );
};

export default DataDisplay;