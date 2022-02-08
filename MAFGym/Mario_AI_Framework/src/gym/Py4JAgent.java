package gym;

import engine.core.MarioAgent;
import engine.core.MarioForwardModel;
import engine.core.MarioTimer;
import engine.helper.MarioActions;

import java.awt.event.KeyEvent;

public class Py4JAgent implements MarioAgent {
    private boolean[] actions = null;

    @Override
    public void initialize(MarioForwardModel model, MarioTimer timer) {
        actions = new boolean[MarioActions.numberOfActions()];
    }

    @Override
    public boolean[] getActions(MarioForwardModel model, MarioTimer timer) {
        return actions;
    }

    @Override
    public String getAgentName() {
        return "Py4JAgent";
    }

    public void setActions(boolean[] newActions) {
        actions = newActions;
    }
}
