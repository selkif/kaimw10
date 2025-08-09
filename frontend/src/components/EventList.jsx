import React from 'react';

const EventList = ({ events }) => {
    return (
        <div>
            <h2>Key Events</h2>
            <ul>
                {events.length > 0 ? (
                    events.map((event, index) => (
                        <li key={index}>{event.EventDescription || event.Event}</li> // Adjust based on your data
                    ))
                ) : (
                    <li>No events available.</li>
                )}
            </ul>
        </div>
    );
};

export default EventList;