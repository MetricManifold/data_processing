import React, { useEffect, useState } from 'react';
import { Typography, Box, CircularProgress, Button, Grid2 } from '@mui/material';
import axios from 'axios';

interface InspectDisplayProps {
    directory: string;
    currentIndex: number;
}

interface CheckpointImageData {
    index: number;
    encodings: string[];
    encodings_full: string[];
    encodings_border: string[];
}

const InspectDisplay: React.FC<InspectDisplayProps> = ({ directory, currentIndex }) => {
    const [images, setImages] = useState<CheckpointImageData[]>([]);
    const [fieldIndex, setFieldIndex] = useState<number>(0);
    const [loading, setLoading] = useState<boolean>(false);
    const [periodicIndex, setPeriodicIndex] = useState<number>(NaN);

    useEffect(() => {
        if (images[fieldIndex]?.encodings.length > 0) {
            setPeriodicIndex(currentIndex % images[fieldIndex]?.encodings.length);
        }
    }, [currentIndex, images]);

    useEffect(() => {
        setLoading(true);
        axios.get(`http://127.0.0.1:3030/get_fields/${directory}`)
            .then(response => {
                const sortedData = response.data.sort((a: CheckpointImageData, b: CheckpointImageData) => a.index - b.index);
                setImages(sortedData);
                setLoading(false);
            })
            .catch(error => {
                console.error('Error fetching images:', error);
                setImages([]);
                setLoading(false);
            });
    }, [directory]);

    // const handleFirstDimensionChange = (event: SelectChangeEvent<number>) => {
    //     setFieldIndex(event.target.value as number);
    // };

    const handleFieldIndexChange = (index: number) => {
        setFieldIndex(index);
    };
    return (
        <Box>
            {loading ? (
                <Box display="flex" justifyContent="center" alignItems="center" height="100%">
                    <CircularProgress />
                </Box>
            ) : (
                images.length > 0 ? (
                    <Box >
                        <Grid2 container justifyContent={"space-between"} >
                            {images.map((frames, index) => (
                                <Button
                                    key={index}
                                    variant={fieldIndex === index ? "contained" : "outlined"}
                                    color="primary"
                                    onClick={() => handleFieldIndexChange(index)}
                                    size="small"
                                    sx={{ padding: 0 }}
                                >
                                    {index}
                                </Button>
                            ))}
                        </Grid2>
                        <Box
                            component="img"
                            src={`data:image/png;base64,${images[fieldIndex].encodings[periodicIndex]}`}
                            alt={`Image ${periodicIndex + 1} for ${directory}`}
                            sx={{ width: '100%', height: 'auto', border: 1 }}
                        />
                        <Box position="relative" width="100%" height="auto" sx={{ paddingTop: '100%', border: 1 }}>
                            {images.map((frames, index) => (
                                <Box
                                    key={index}
                                    component="img"
                                    src={`data:image/png;base64,${frames.encodings_full[periodicIndex]}`}
                                    alt={`Image ${index + 1} for ${directory}`}
                                    sx={{
                                        position: 'absolute',
                                        top: 0,
                                        left: 0,
                                        width: '100%',
                                        height: '100%',
                                        objectFit: 'contain',
                                    }}
                                />
                            ))}
                            <Box
                                key={images.length}
                                component="img"
                                src={`data:image/png;base64,${images[fieldIndex].encodings_border[periodicIndex]}`}
                                alt={`Image ${images.length + 1} for ${directory}`}
                                sx={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    width: '100%',
                                    height: '100%',
                                    objectFit: 'contain',
                                }}
                            />
                        </Box>
                    </Box>
                ) : (
                    <Typography variant="body1">No images found for {directory}</Typography>
                )
            )}
        </Box>
    );
};

export default InspectDisplay;